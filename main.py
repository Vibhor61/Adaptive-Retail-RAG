"""
Main FastAPI application module for the RAG pipeline.
Handles API routing, session management, and lifecycle events.
Initializes language models and orchestration graphs.
Provides endpoints for standard chat and evaluation chat workflows.
"""
from typing import Any, List, Optional
from uuid import uuid4
import os
import threading
import json
import traceback
import queue
import psycopg2

from contextlib import asynccontextmanager
from pydantic import BaseModel
from opentelemetry import trace
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from config.settings import settings
from config.telemetry import setup_tracing, shutdown_tracing
from graph_layer.graph import build_graph
from eval.eval_graph import build_eval_graph

from orchestration.router_orchestrator import RouterOrchestrator
from orchestration.generation_orchestrator import GenerationOrchestrator

tracer = trace.get_tracer(__name__)

def get_db_connection():
    return psycopg2.connect(settings.postgres_url)


compiled_graph = None
compiled_eval_graph = None
generation_llm_global = None
router_llm_global = None
rewrite_llm_global = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Initializes tracing, language models, and computational graphs on startup, and shuts down tracing on exit.
    """
    global compiled_graph, compiled_eval_graph, generation_llm_global, router_llm_global, rewrite_llm_global

    # 1. Tracing must come first
    setup_tracing()

    # Create sessions_table if not exists
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions_table (
                    session_id TEXT PRIMARY KEY,
                    history JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ DEFAULT now()
                );
            """)
            conn.commit()

    # 2. Extract separate keys for static split
    keys_str = os.environ.get("GROQ_API_KEYS") or settings.groq_api_key
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    key_router_rewrite = keys[0] if len(keys) > 0 else settings.groq_api_key
    key_generation = keys[1] if len(keys) > 1 else key_router_rewrite

    # 3. LLMs instantiated here
    generation_llm_global = ChatGroq(
        model=settings.groq_model_name,
        temperature=0,
        api_key=key_generation,
        streaming=True,
    )
    router_llm_global = ChatGroq(
        model=settings.router_model,
        temperature=0,
        api_key=key_router_rewrite,
    )
    rewrite_llm_global = ChatGroq(
        model=settings.rewrite_model,
        temperature=0,
        api_key=key_router_rewrite,
    )

    compiled_graph = build_graph(
        generation_llm=generation_llm_global,
        router_llm=router_llm_global,
        rewrite_llm=rewrite_llm_global,
    )

    compiled_eval_graph = build_eval_graph(
        router=RouterOrchestrator(router_llm_global),
        generation=GenerationOrchestrator(generation_llm_global),
    )

    yield
    
    shutdown_tracing()


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)

"""
Pydantic Request/Response Models
"""

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: Optional[str]
    citations: List[Any] = []
    system_failure: Optional[str] = None


class EvalChatRequest(BaseModel):
    query: str


class EvalControlInfo(BaseModel):
    stage: Optional[str] = None
    should_clarify: bool = False
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None


class EvalChatResponse(BaseModel):
    answer: Optional[str]
    citations: List[Any] = []
    control: EvalControlInfo
    router_result: Optional[Any] = None
    router_guardrails: Optional[Any] = None
    retrieval_result: Optional[Any] = None
    retrieval_guardrails: Optional[Any] = None
    generation_result: Optional[Any] = None


class StreamingCallback(BaseCallbackHandler):
    def __init__(self, q: queue.Queue):
        self.q = q
        self._generation_active = False

    def start_generation(self) -> None:
        self._generation_active = True

    def stop_generation(self) -> None:
        self._generation_active = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self._generation_active:
            self.q.put(token)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Handles streaming chat requests.
    Returns Server-Sent Events (SSE) of the generated answer tokens.
    """
    if compiled_graph is None:
        raise HTTPException(status_code=500, detail="graph_not_initialized")

    session_id = request.session_id or str(uuid4())

    def generate():
        q = queue.Queue()
        cb = StreamingCallback(q)
        config = {"callbacks": [cb]}

        def run_graph():
            try:
                # Load history from PostgreSQL
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT history FROM sessions_table WHERE session_id = %s;", (session_id,))
                        row = cur.fetchone()
                        if row:
                            history = row[0]
                            if isinstance(history, str):
                                history = json.loads(history)
                        else:
                            history = []

                initial_state = {
                    "query": {
                        "original_query": request.query,
                        "rewritten_query": None,
                    },
                    "chat_history": history,
                }

                with tracer.start_as_current_span("main_pipeline_stream"):
                    final_state = compiled_graph.invoke(initial_state, config=config)

                    control = final_state.get("control", {}) or {}
                    response = final_state.get("response", {}) or {}
                    did_clarify = control.get("should_clarify", False)
                    answer = response.get("answer")

                    if not did_clarify and answer:
                        history.append({"role": "user", "content": request.query})
                        history.append({"role": "assistant", "content": answer})
                        # Save history back to PostgreSQL
                        with get_db_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    INSERT INTO sessions_table (session_id, history)
                                    VALUES (%s, %s::jsonb)
                                    ON CONFLICT (session_id)
                                    DO UPDATE SET history = EXCLUDED.history, updated_at = now();
                                """, (session_id, json.dumps(history)))
                                conn.commit()
                            
                            if did_clarify or "generation" not in final_state:
                                if answer:
                                    q.put(answer)
                                    
                            retrieved_contexts = []
                            retrieval_output = final_state.get("retrieval", {}).get("output")
                            if retrieval_output and hasattr(retrieval_output, "evaluation_bundles"):
                                for bundle in retrieval_output.evaluation_bundles:
                                    if bundle.bundle and hasattr(bundle.bundle, "items"):
                                        for item in bundle.bundle.items:
                                            retrieved_contexts.append(item.text)
                                            
                            cites = []
                            for c in response.get("citations", []):
                                if hasattr(c, "model_dump"):
                                    cites.append(c.model_dump())
                                elif hasattr(c, "dict"):
                                    cites.append(c.dict())
                                else:
                                    cites.append(str(c))
                                    
                            final_metadata = {
                                "type": "metadata",
                                "control": control,
                                "citations": cites,
                                "retrieved_contexts": retrieved_contexts,
                                "session_id": session_id,
                                "system_failure": control.get("failure_reason") if did_clarify else None
                            }
                            q.put(final_metadata)
            except Exception as exc:
                traceback.print_exc()
                q.put({"type": "error", "error": type(exc).__name__, "detail": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=run_graph).start()

        while True:
            try:
                token = q.get(timeout=180)
            except queue.Empty:
                break
            if token is None:
                break
            yield f"data: {json.dumps(token)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str) -> dict:
    """
    Deletes a specific session from the session store.
    Returns a dictionary confirming the deletion of the session ID.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions_table WHERE session_id = %s;", (session_id,))
            conn.commit()

    return {"cleared": session_id}


@app.post("/eval_chat", response_model=EvalChatResponse)
def eval_chat(request: EvalChatRequest) -> EvalChatResponse:
    """
    Processes chat requests specifically for evaluation purposes.
    Runs the query through the evaluation graph and returns detailed pipeline telemetry and state.
    """

    if compiled_eval_graph is None:
        raise HTTPException(status_code=500, detail="eval_graph_not_initialized")

    # No rewrite node in this graph: feed the raw query straight into
    initial_state = {
        "query": {
            "original_query": request.query,
            "rewritten_query": request.query,
        },
        "chat_history": [],
    }

    with tracer.start_as_current_span("eval_pipeline") as span:
        span.set_attribute("query", request.query)

        try:
            final_state = compiled_eval_graph.invoke(initial_state)

            if final_state is None:
                raise RuntimeError("graph_returned_none")

            control = final_state.get("control", {}) or {}
            response = final_state.get("response", {}) or {}
            router_state = final_state.get("router", {}) or {}
            retrieval_state = final_state.get("retrieval", {}) or {}
            generation_state = final_state.get("generation", {}) or {}

            span.set_attribute("status", "failed" if control.get("should_clarify") else "success")

            return EvalChatResponse(
                answer=response.get("answer"),
                citations=response.get("citations", []),
                control=EvalControlInfo(
                    stage=control.get("stage"),
                    should_clarify=control.get("should_clarify", False),
                    failure_stage=control.get("failure_stage"),
                    failure_reason=control.get("failure_reason"),
                ),
                router_result=router_state.get("output"),
                router_guardrails=router_state.get("guardrails"),
                retrieval_result=retrieval_state.get("output"),
                retrieval_guardrails=retrieval_state.get("guardrails"),
                generation_result=generation_state.get("output"),
            )

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("status", "failed")

            return EvalChatResponse(
                answer=None,
                citations=[],
                control=EvalControlInfo(
                    stage=None,
                    should_clarify=True,
                    failure_stage="pipeline",
                    failure_reason=str(e),
                ),
                router_result=None,
                router_guardrails=None,
                retrieval_result=None,
                retrieval_guardrails=None,
                generation_result=None,
            )


@app.get("/health")
def health() -> dict:
    """
    Health check endpoint to verify the API status.
    Returns the initialization status of the computational graphs.
    """
    return {
        "status": "ok",
        "graph_ready": compiled_graph is not None,
        "eval_graph_ready": compiled_eval_graph is not None,
    }
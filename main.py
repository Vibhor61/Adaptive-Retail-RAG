from typing import Any, List, Optional
from uuid import uuid4
import shelve
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from opentelemetry import trace

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from config.settings import settings
from config.telemetry import setup_tracing
from graph_layer.graph import build_graph
from eval.eval_graph import build_eval_graph

from orchestration.router_orchestrator import RouterOrchestrator
from orchestration.generation_orchestrator import GenerationOrchestrator

tracer = trace.get_tracer(__name__)

SESSION_DB_PATH = "data/sessions.db"
os.makedirs("data", exist_ok=True)

_shelve_lock = threading.Lock()


def load_session_store():
    return shelve.open(SESSION_DB_PATH, writeback=True)


compiled_graph = None
compiled_eval_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph, compiled_eval_graph

    # 1. Tracing must come first
    setup_tracing()

    # 2. LLMs instantiated here, after tracing is ready
    generation_llm = ChatGroq(
        model=settings.groq_model_name,
        temperature=0,
        api_key=settings.groq_api_key,
    )
    router_llm = ChatOllama(
        model=settings.router_model,
        temperature=0,
        base_url=settings.ollama_url,
    )
    rewrite_llm = ChatOllama(
        model=settings.rewrite_model,
        temperature=0,
        base_url=settings.ollama_url,
    )

    compiled_graph = build_graph(
        generation_llm=generation_llm,
        router_llm=router_llm,
        rewrite_llm=rewrite_llm,
    )

    compiled_eval_graph = build_eval_graph(
        router=RouterOrchestrator(router_llm),
        generation=GenerationOrchestrator(generation_llm),
    )

    yield

    # Shutdown
    from config.telemetry import shutdown_tracing
    shutdown_tracing()


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: Optional[str]
    citations: List[Any] = []
    system_failure: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:

    if compiled_graph is None:
        raise HTTPException(status_code=500, detail="graph_not_initialized")

    session_id = request.session_id or str(uuid4())

    with _shelve_lock:
        with load_session_store() as db:
            history = db.get(session_id, [])

            initial_state = {
                "query": {
                    "original_query": request.query,
                    "rewritten_query": None,
                },
                "chat_history": history,
            }

            with tracer.start_as_current_span("main_pipeline") as span:
                span.set_attribute("query", request.query)
                span.set_attribute("session_id", session_id)

                try:
                    final_state = compiled_graph.invoke(initial_state)
                except Exception as exc:
                    span.set_attribute("pipeline.status", "failed")
                    span.record_exception(exc)
                    raise HTTPException(status_code=500, detail="pipeline_failed") from exc

                control = final_state.get("control", {}) or {}
                response = final_state.get("response", {}) or {}

                did_clarify = control.get("should_clarify", False)
                failure_reason = control.get("failure_reason")

                answer = response.get("answer")
                citations = response.get("citations", [])

                span.set_attribute(
                    "pipeline.status",
                    "failed" if did_clarify else "success"
                )
                if not did_clarify:
                    history.append({"role": "user", "content": request.query})
                    history.append({"role": "assistant", "content": answer})
                    db[session_id] = history
                    db.sync()

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        citations=citations,
        system_failure=failure_reason if did_clarify else None,
    )


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str) -> dict:
    with _shelve_lock:
        with load_session_store() as db:
            if session_id in db:
                del db[session_id]
                db.sync()

    return {"cleared": session_id}


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


@app.post("/eval_chat", response_model=EvalChatResponse)
def eval_chat(request: EvalChatRequest) -> EvalChatResponse:

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
    return {
        "status": "ok",
        "graph_ready": compiled_graph is not None,
        "eval_graph_ready": compiled_eval_graph is not None,
    }
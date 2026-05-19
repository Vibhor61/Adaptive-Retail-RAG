from langgraph.graph import StateGraph, END
from typing import List, Optional, Annotated, TypedDict
import operator
import uuid
from router import route, RouterStatus
from retrieval import retreive, RetrievalStatus, RetrievalResult

from scripts.generation import build_prompt, generate_answer, AnswerStatus
from rewrite import rewrite_query 
from evaluation import evaluate_answer
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class State(TypedDict):
    """State schema for the RAG graph."""
    # Run Data
    request_id: str
    history: Annotated[list[str], operator.add]
    current_query: str
    
    # Routing state
    routing_decision: str
    routing_confidence: float

    # Retrieval state
    retrieval_type: str
    postgres_data: Optional[List[dict]]
    qdrant_data: Optional[List[dict]]
    retrieval_score: float
    retrieval_retries: int
    retrieval_valid: bool

    # Answer State
    answer: str
    model_used: str
    answer_score: float
    retries: int

    # Final Graph State
    attempt: int
    attempt_history: List[dict]

    failure_stage: Optional[str]
    failure_type: Optional[str]
    final_status: Optional[str]


    error_message: Optional[str]
    user_satisfied: Optional[bool]
    should_clarify: Optional[bool]
    model_index: int


def start_node(state: State):
    request_id = str(uuid.uuid4())

    span = tracer.start_span(
        "rag_request",
        attributes={
            "request.id":request_id,
            "query": state["current_query"]
        }
    )

    ctx = trace.set_span_in_context(span)

    return{
        "request_id":request_id,
        "_trace_span": span,
        "_trace_ctx": ctx
    }


def router_node(state: State) -> dict:
    """Route the query to appropriate retrieval strategy."""
    
    with tracer.start_as_current_span("node_name", context=state["_trace_ctx"]) as span:
        try:
            result = route(state["current_query"])
            
            span.set_attribute("router.decision", result.retrieval_type)
            span.set_attribute("router.confidence", result.confidence)
            span.set_attribute("router.status", result.status.value)
            span.set_attribute("router.reason", result.reason)
            span.set_attribute("router.llm_used", result.llm_used)

            signals = result.signals 
            span.set_attribute("router.signal_type", signals.get("type", "unknown"))
            
            if result.failure_reason:
                span.set_attribute("router.failure_reason", result.failure_reason)

            output = {
                "routing_decision": result.retrieval_type,
                "routing_confidence": result.confidence,
                "routing_status": result.status.value,
                "routing_reason": result.reason,
                "routing_signals": result.signals,
                "routing_llm_used": result.llm_used
            }

            if result.status != RouterStatus.Passed:
                output.update({
                    "failure_stage": "router",
                    "failure_type": result.status.value,
                    "should_clarify": result.status == RouterStatus.Vague
                })
            
            return output
        except Exception as e:
            span.record_exception(e)
            return {
                "routing_decision": "hybrid",
                "routing_confidence": 0.5,
                "routing_status": result.status.value,
                "error_message": f"Routing failed: {str(e)}"
            }



def retrieve_node(state: State) -> dict:
    """Execute retrieval based on routing decision."""
    
    with tracer.start_as_current_span("node_name", context=state["_trace_ctx"]) as span:
        query = state["current_query"]
        decision = state["routing_decision"]
        try:
            final = retreive(query, decision)
            span.set_attribute("retrieval.mode", decision)
            span.set_attribute("retrieval.status", final.status.value)

            if final.failure_reason:
                span.set_attribute("retrieval.failure_reason", final.failure_reason)

            if final.status != RetrievalStatus.PASSED:
                return {
                    "retrieval_type": decision,
                    "retrieved_data": [],
                    "retrieval_score": 0.0,
                    "retrieval_valid": False,
                    "failure_stage": "retrieval",
                    "failure_type": final.status.value,
                    "retrieval_signals": vars(final.signals) if final.signals else {},
                    "error_message": final.failure_reason
                }

            return {
                "retrieval_type": decision,
                "retrieved_data": final.items,
                "retrieval_score": final.items[0].score if final.items else 0.0,
                "retrieval_valid": True,
                "retrieval_signals": vars(final.signals) if final.signals else {}
            }
        except Exception as e:
            span.record_exception(e)
            return {
                "retrieval_type": decision,
                "retrieved_data": [],
                "retrieval_score": 0.0,
                "retrieval_valid": False,
                "failure_stage": "retrieval",
                "failure_type": "exception",
                "error_message": f"Retrieval failed: {str(e)}"
            }


def generate_node(state: State) -> dict:
    """Generate answer from retrieved data."""
    with tracer.start_as_current_span("generate_node", context=state["_trace_ctx"]) as span:
        query = state["current_query"]
        retrieved_data = state.get("retrieved_data", [])
        model_level = state.get("model_level", 1)
        
        try:
            if not retrieved_data:
                span.set_attribute("generation.status", "skipped")
                span.set_attribute("generation.reason", "no_context")

                return {
                    "answer": "",
                    "answer_valid": False,
                    "answer_score": 0.0,
                    "answer_status": AnswerStatus.FAILED.value,
                    "model_used": None,
                    "error_message": "No context available for generation"
                }

            # Deserialize
            items = [RetrievalResult(**item) for item in retrieved_data]

            postgres_results = [i for i in items if i.source == "postgres"]
            qdrant_results = [i for i in items if i.source == "qdrant"]

            prompt = build_prompt(query, postgres_results, qdrant_results)

            answer_text, model_name = generate_answer(prompt, model_level)

            span.set_attribute("generation.model", model_name)
            span.set_attribute("generation.length", len(answer_text))

            if not answer_text:
                span.set_attribute("generation.status", "failed")

                return {
                    "answer": "",
                    "answer_valid": False,
                    "answer_score": 0.0,
                    "answer_status": AnswerStatus.FAILED.value,
                    "model_used": model_name,
                    "error_message": "LLM generation failed"
                }

            span.set_attribute("generation.status", "success")

            return {
                "answer": answer_text,
                "answer_valid": True,
                "answer_score": 1.0,
                "answer_status": AnswerStatus.GENERATED.value,
                "model_used": model_name
            }

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("generation.status", "error")

            return {
                "answer": "",
                "answer_valid": False,
                "answer_score": 0.0,
                "answer_status": AnswerStatus.FAILED.value,
                "model_used": None,
                "error_message": f"Generation failed: {str(e)}"
            }



def evaluate_node(state: State) -> dict:
    """Evaluate generated answer quality."""

    with tracer.start_as_current_span("evaluate_node", context=state["_trace_ctx"]) as span:
        try:
            answer = state.get("answer", "")
            retrieved_data = state.get("retrieved_data", [])

            # Guard: no answer → cannot evaluate
            if not answer:
                span.set_attribute("evaluation.status", "skipped")
                span.set_attribute("evaluation.reason", "no_answer")

                return {
                    "answer_score": 0.0,
                    "evaluation_valid": False
                }

            context = "\n".join([str(item) for item in retrieved_data])

            evaluation = evaluate_answer(
                state["current_query"],
                answer,
                context
            )

            score = evaluation.get("score", 0.0)

            span.set_attribute("evaluation.score", score)
            span.set_attribute("evaluation.status", "success")

            return {
                "answer_score": score,
                "evaluation_valid": True
            }

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("evaluation.status", "error")

            return {
                "answer_score": 0.0,
                "evaluation_valid": False,
                "error_message": f"Evaluation failed: {str(e)}"
            }



def rewrite_node(state: State) -> dict:
    """Rewrite query using chat history."""

    with tracer.start_as_current_span("rewrite_node", context=state["_trace_ctx"]) as span:
        try:
            original_query = state["current_query"]
            history = state.get("history", [])

            rewritten = rewrite_query(original_query, history)

            span.set_attribute("rewrite.original_length", len(original_query))
            span.set_attribute("rewrite.new_length", len(rewritten))
            span.set_attribute("rewrite.status", "success")

            return {
                "current_query": rewritten,
                "retrieval_retries": state.get("retrieval_retries", 0) + 1
            }

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("rewrite.status", "error")

            return {
                "current_query": state["current_query"],  # fallback
                "retrieval_retries": state.get("retrieval_retries", 0) + 1,
                "error_message": f"Rewrite failed: {str(e)}"
            }


def escalate_node(state: State) -> dict:
    """Escalate to a better model on failure."""

    with tracer.start_as_current_span("escalate_node", context=state["_trace_ctx"]) as span:
        try:
            current_index = state.get("model_index", 1)
            next_index = min(current_index + 1, 3)

            span.set_attribute("escalation.from", current_index)
            span.set_attribute("escalation.to", next_index)
            span.set_attribute("escalation.status", "success")

            return {
                "model_index": next_index,
                "escalation_applied": True
            }

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("escalation.status", "error")

            return {
                "model_index": state.get("model_index", 1),
                "escalation_applied": False,
                "error_message": f"Escalation failed: {str(e)}"
            }

def controller_node(state: State) -> dict:
    with tracer.start_as_current_span("controller_node", context=state["_trace_ctx"]) as span:
        
        if not state.get("failure_stage"):
            return {
                "next_step":"router"
            }
        
        stage = state.get("failure_stage")

        if stage == "router":
            span.set_attribute("controller.stage","router")
            status = state.get("failure_type")
            if status == "vague":
                span.set_attribute("controller.status","vague")
                span.set_attribute("controller.next_step","clarify")
                return {
                    "should_clarify": True,
                    "next_step": "clarify"
                }

            # Ambiguous → force hybrid
            if status == "ambiguous":
                span.set_attribute("controller.status","ambiguous")
                span.set_attribute("controller.next_step","retrieval")
                return {
                    "routing_decision": "hybrid",
                    "next_step": "retrieval"
                }

            if status == "low_confidence":
                span.set_attribute("controller.status","low_confidence")
                span.set_attribute("controller.next_step","rewrite")
                return {
                    "next_step": "rewrite"
                }
            
        if stage == "retrieval":
            retries = state.get("retrieval_retries", 0)
            failure_type = state.get("failure_type")
            current_mode = state.get("retrieval_type")

            # Retry limit
            if retries >= 2:
                return {
                    "next_step": "no_data",
                    "error_message": "No sufficient data found after retries"
                }

            if failure_type == "empty":
                if current_mode == "sparse":
                    next_mode = "dense"
                elif current_mode == "dense":
                    next_mode = "hybrid"
                else:
                    next_mode = "sparse"

                return {
                    "routing_decision": next_mode,
                    "retrieval_retries": retries + 1,
                    "next_step": "retrieval"
                }

            # LOW QUALITY → escalate
            if failure_type == "low_quality":

                if current_mode != "hybrid":
                    return {
                        "routing_decision": "hybrid",
                        "retrieval_retries": retries + 1,
                        "next_step": "retrieval"
                    }
                else:
                    return {
                        "next_step": "no_data",
                        "error_message": "No sufficient data found after retries"
                    }
                
            # SEMANTIC FAILURE → fallback to sparse
            if failure_type == "semantic_failure":
                return {
                    "routing_decision": "sparse",
                    "retrieval_retries": retries + 1,
                    "next_step": "retrieval"
                }
            
        if stage == "generation":
            span.set_attribute("controller.stage", "generation")
            failure_type = state.get("failure_type")
            model_index = state.get("model_index", 1)
            retrieval_mode = state.get("retrieval_type", "hybrid")

            span.set_attribute("controller.status", failure_type)
            span.set_attribute("controller.model_index", model_index)

            # Try a stronger model first
            if model_index < 3:
                span.set_attribute("controller.next_step", "generate")
                return {
                    "model_index": model_index + 1,
                    "next_step": "generate"
                }

            # If retrieval was not hybrid, broaden the context and retry
            if retrieval_mode != "hybrid":
                span.set_attribute("controller.next_step", "retrieval")
                return {
                    "routing_decision": "hybrid",
                    "next_step": "retrieval"
                }

            span.set_attribute("controller.next_step", "rewrite")
            return {
                "next_step": "rewrite"
            }


        
def clarify_node(state: State) -> dict:
    
    """Handle low confidence routing by asking for clarification."""

    with tracer.start_as_current_span("clarify_node", context=state["_trace_ctx"]) as span:
        try:
            span.set_attribute("clarify.status", "triggered")

            return {
                "should_clarify": True,
                "clarification_reason": "ambiguous_query",
                "answer_valid": False
            }

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("clarify.status", "error")

            return {
                "should_clarify": True,
                "answer_valid": False,
                "error_message": f"Clarify node failed: {str(e)}"
            }
from contextlib import contextmanager
from opentelemetry import trace

from contracts.router_contracts import Intent, ValidityStatus
from contracts.orchestration_contracts import (
    RouterLayerOutput,
    RetrievalLayerOutput,
    GenerationLayerOutput,
)

from retrieval_layer.retrieval_guardrails import run_retrieval_structural_guardrails
from routing_layer.router_guardrails import run_structural_guardrails

tracer = trace.get_tracer(__name__)


@contextmanager
def trace_node(name: str, attributes: dict = None):
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        try:
            yield span
            span.set_attribute("status", "success")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("status", "failed")
            raise


def _control(stage, should_clarify, failure_stage=None, failure_reason=None):
    return {
        "control": {
            "stage": stage,
            "should_clarify": should_clarify,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
        }
    }


def make_rewrite_node(rewrite_llm):
    def rewrite_node(state):
        query = state["query"]["original_query"]
        history = state.get("chat_history", [])

        with trace_node("rewrite_node", {"query": query}):
            if not history:
                return {
                    "query": {"rewritten_query": query},
                    **_control(stage="rewrite", should_clarify=False),
                }

            prompt = f"""
                Rewrite the query for retrieval based on conversation history.
                Return only the rewritten query.

                Query:
                {query}

                History:
                {history}
            """

            response = rewrite_llm.invoke(prompt)
            return {
                "query": {"rewritten_query": response.content.strip()},
                **_control(stage="rewrite", should_clarify=False),
            }

    return rewrite_node


def make_router_node(router):
    def router_node(state):
        query = state["query"]["rewritten_query"] or state["query"]["original_query"]

        with trace_node("router_node", {"query": query}):
            router_result: RouterLayerOutput = router.run(query)

            system_failed = router_result.system_failure is not None

            return {
                "router": {
                    "output": router_result,
                    "guardrails": None,
                },
                **_control(
                    stage="route",
                    should_clarify=system_failed,
                    failure_stage="router" if system_failed else None,
                    failure_reason=(
                        router_result.system_failure.message if router_result.system_failure else None
                    ),
                ),
            }

    return router_node


def router_guardrail_node(state):
    router_result: RouterLayerOutput = state["router"]["output"]

   
    if router_result.system_failure is not None:
        return {
            "router": {
                "output": router_result,
                "guardrails": None,
            },
            **_control(
                stage="route",
                should_clarify=True,
                failure_stage="router",
                failure_reason=router_result.system_failure.message,
            ),
        }

    with trace_node("router_guardrail_node"):
        guardrail_result = run_structural_guardrails(router_result.router_output)

        validity_failed = router_result.validity_result.status == ValidityStatus.DEGRADED
        guardrail_failed = not guardrail_result.passed
        should_clarify = validity_failed or guardrail_failed

        if validity_failed:
            failure_reason = "query_validity_failed"
        elif guardrail_failed:
            failure_reason = "router_structural_guardrail_failed"
        else:
            failure_reason = None

        return {
            "router": {
                "output": router_result,
                "guardrails": guardrail_result,
            },
            **_control(
                stage="route",
                should_clarify=should_clarify,
                failure_stage="router" if should_clarify else None,
                failure_reason=failure_reason,
            ),
        }



def post_router_node(state):
    router: RouterLayerOutput = state["router"]["output"]
    upstream_should_clarify = state["control"]["should_clarify"]
    upstream_failure_stage = state["control"]["failure_stage"]
    upstream_failure_reason = state["control"]["failure_reason"]

    if upstream_should_clarify:
        return _control(
            stage="route",
            should_clarify=True,
            failure_stage=upstream_failure_stage,
            failure_reason=upstream_failure_reason,
        )

    if router.router_output is None:
        return _control(
            stage="route",
            should_clarify=True,
            failure_stage="router",
            failure_reason="missing_router_output",
        )

    if router.router_output.intent_type == Intent.UNKNOWN:
        return _control(
            stage="route",
            should_clarify=True,
            failure_stage="router",
            failure_reason="unknown_intent",
        )

    if router.router_output.intent_type in (Intent.LOOKUP, Intent.COMPARISON):
        if not router.grounded_entities:
            return _control(
                stage="route",
                should_clarify=True,
                failure_stage="router",
                failure_reason="missing_entities",
            )

    return _control(stage="retrieve", should_clarify=False)


def retrieval_node(state):
    router_output: RouterLayerOutput = state["router"]["output"]

    with trace_node("retrieval_node"):
        from orchestration.retrieval_orchestrator import run_retrieval_pipeline

        retrieval_result: RetrievalLayerOutput = run_retrieval_pipeline(router_output)

        system_failed = retrieval_result.system_failure is not None

        return {
            "retrieval": {
                "output": retrieval_result,
                "guardrails": None,
            },
            **_control(
                stage="retrieve",
                should_clarify=system_failed,
                failure_stage="retrieval" if system_failed else None,
                failure_reason=(
                    retrieval_result.system_failure.message
                    if retrieval_result.system_failure
                    else None
                ),
            ),
        }


def retrieval_guardrail_node(state):
    retrieval: RetrievalLayerOutput = state["retrieval"]["output"]

    if retrieval.system_failure is not None:
        return _control(
            stage="retrieve",
            should_clarify=True,
            failure_stage="retrieval",
            failure_reason=retrieval.system_failure.message,
        )

    with trace_node("retrieval_guardrail_node"):
        guardrail_result = run_retrieval_structural_guardrails(retrieval.evaluation_bundles)
        should_clarify = not guardrail_result.passed

        return {
            "retrieval": {
                "output": retrieval,
                "guardrails": guardrail_result,
            },
            **_control(
                stage="generate" if not should_clarify else "retrieve",
                should_clarify=should_clarify,
                failure_stage="retrieval" if should_clarify else None,
                failure_reason="retrieval_structural_guardrail_failed" if should_clarify else None,
            ),
        }



def make_generation_node(generation):
    def generation_node(state):
        retrieval: RetrievalLayerOutput = state["retrieval"]["output"]

        with trace_node("generation_node"):
            generation_result: GenerationLayerOutput = generation.run(retrieval)

            should_clarify = generation_result.system_failure is not None

            return {
                "generation": {
                    "output": generation_result,
                },
                "response": {
                    "answer": generation_result.answer,
                    "citations": generation_result.citations,
                },
                **_control(
                    stage="end" if not should_clarify else "generate",
                    should_clarify=should_clarify,
                    failure_stage="generation" if should_clarify else None,
                    failure_reason=(
                        generation_result.system_failure.message
                        if generation_result.system_failure
                        else None
                    ),
                ),
            }

    return generation_node



_CLARIFICATION_MESSAGES = {
    ("router", "query_validity_failed"):
        "I couldn't quite parse that as a valid query could you rephrase it?",
    ("router", "router_structural_guardrail_failed"):
        "The query can't be understood completely could you rephrase it?",
    ("router", "unknown_intent"):
        "I'm not sure what you're asking for could you clarify what you'd like to know?",
    ("router", "missing_entities"):
        "I couldn't identify what you're referring to could you specify it more directly?",
    ("router", "missing_router_output"):
        "Something went wrong while processing your query. Could you try rephrasing it?",
    ("retrieval", "retrieval_structural_guardrail_failed"):
        "I couldn't find enough relevant information to answer that confidently.",
}

_DEFAULT_MESSAGE = "Please rephrase your query."


def clarification_node(state):
    control = state["control"]
    failure_stage = control.get("failure_stage")
    failure_reason = control.get("failure_reason")

    if failure_stage in ("router", "retrieval") and (failure_stage, failure_reason) in _CLARIFICATION_MESSAGES:
        message = _CLARIFICATION_MESSAGES[(failure_stage, failure_reason)]
    elif failure_stage is not None and failure_reason is not None:
        message = "Something went wrong while handling your query. Could you try again?"
    else:
        message = _DEFAULT_MESSAGE

    return {
        "response": {
            "answer": message,
        },
        **_control(
            stage="clarify",
            should_clarify=True,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
        ),
    }


def route_after_router_guardrail(state):
    return "clarify" if state["control"]["should_clarify"] else "post_router"


def route_after_post_router(state):
    return "clarify" if state["control"]["should_clarify"] else "retrieve"


def route_after_retrieval_guardrail(state):
    return "clarify" if state["control"]["should_clarify"] else "generate"


def route_after_generation(state):
    return "clarify" if state["control"]["should_clarify"] else "__end__"
import logging

from opentelemetry import trace

from contracts.orchestration_contracts import (
    RouterLayerOutput,
    RetrievalLayerOutput,
    ExceptionInfo
)

from contracts.router_contracts import (
    Intent
)

from contracts.retrieval_contracts import (
    RetrievalPlan,
)

from retrieval_layer.retrieval_strategies import (
    lookup_workflow,
    comparison_workflow,
    recommendation_workflow
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

def make_retrieval_plan(input: RouterLayerOutput)->RetrievalPlan:

    return RetrievalPlan(
        original_query =input.normalized_query,
        intent_type=input.router_output.intent_type,
        evidence_type=input.router_output.evidence_type,
        grounded_entities=input.grounded_entities,
        entity_structure=input.router_output.entity_structure
    )

def run_retrieval_pipeline(input: RouterLayerOutput) -> RetrievalLayerOutput:
    
    with tracer.start_as_current_span("retrieval_pipeline") as span:

        span.set_attribute("retrieval.query",input.normalized_query,)

        try:

            plan = make_retrieval_plan(input)

            if plan.intent == Intent.LOOKUP:
                output = lookup_workflow(plan)

            elif plan.intent == Intent.COMPARISON:
                output = comparison_workflow(plan)

            elif plan.intent == Intent.RECOMMENDATION:
                output = recommendation_workflow(plan)

            else:
                raise NotImplementedError(
                    f"intent '{plan.intent}' not supported"
                )

            span.set_attribute("retrieval.status", "success")

            return output

        except Exception as e:

            span.record_exception(e)
            span.set_attribute("retrieval.status", "error")

            logger.exception("Infrastructure failure in retrieval pipeline")

            return RetrievalLayerOutput(
                plan=locals().get("plan"),
                evaluation_bundles=[],
                system_failure=ExceptionInfo(
                    exception_type=type(e).__name__,
                    message=str(e),
                ),
            )
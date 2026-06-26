"""
Retrieval orchestrator module.
Coordinates different retrieval workflows based on the identified intent type.
Converts router outputs into retrieval plans and executes the appropriate retrieval strategy.
"""

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
    """
    Creates a retrieval plan from the router output.
    Extracts query, intent, evidence type, and entities to guide retrieval.
    """

    return RetrievalPlan(
        original_query=input.normalized_query,
        intent_type=input.router_output.intent_type,
        evidence_type=input.router_output.evidence_type,
        grounded_entities=input.grounded_entities,
        entity_structure=input.router_output.entity_structure
    )

def run_retrieval_pipeline(input: RouterLayerOutput) -> RetrievalLayerOutput:
    """
    Executes the retrieval pipeline for a given router output.
    Dispatches to lookup, comparison, or recommendation workflows based on intent,
    handling and logging any retrieval failures.
    """
    
    with tracer.start_as_current_span("retrieval_pipeline") as span:

        span.set_attribute("retrieval.query",input.normalized_query,)

        try:

            plan = make_retrieval_plan(input)

            if plan.intent_type == Intent.LOOKUP:
                output = lookup_workflow(plan)

            elif plan.intent_type == Intent.COMPARISON:
                output = comparison_workflow(plan)

            elif plan.intent_type == Intent.RECOMMENDATION:
                output = recommendation_workflow(plan)

            else:
                raise NotImplementedError(
                    f"intent '{plan.intent_type}' not supported"
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
from opentelemetry import trace

from orchestration.router_orchestrator import run_router_pipeline
from orchestration.retrieval_orchestrator import run_retrieval_pipeline
from orchestration.generation_orchestrator import run_generation_pipeline

tracer = trace.get_tracer(__name__)


def controller(query: str):

    with tracer.start_as_current_span("main_pipeline") as span:

        span.set_attribute("query", query)

        router_result = run_router_pipeline(query)

        if router_result.system_failure:
            span.set_attribute("pipeline.stage", "router")
            span.set_attribute("pipeline.status", "failed")
            return router_result

        retrieval_result = run_retrieval_pipeline(router_result)

        if retrieval_result.system_failure:
            span.set_attribute("pipeline.stage", "retrieval")
            span.set_attribute("pipeline.status", "failed")
            return retrieval_result

        generation_result = run_generation_pipeline(retrieval_result)

        if generation_result.system_failure:
            span.set_attribute("pipeline.stage", "generation")
            span.set_attribute("pipeline.status", "failed")
        else:
            span.set_attribute("pipeline.status", "success")

        return generation_result
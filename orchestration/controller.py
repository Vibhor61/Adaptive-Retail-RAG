from dataclasses import dataclass
from typing import Any, Optional, List
from opentelemetry import trace

from orchestration.router_orchestrator import RouterOrchestrator
from orchestration.retrieval_orchestrator import run_retrieval_pipeline
from orchestration.generation_orchestrator import run_generation_pipeline
from config.telemetry import setup_tracing
from config.settings import settings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

setup_tracing()

tracer = trace.get_tracer(__name__)

generation_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=settings.groq_api_key,
)

router_llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
    base_url="http://127.0.0.1:5105",
)
router = RouterOrchestrator(llm=router_llm)


@dataclass
class ControllerResult:
    system_failure: Optional[str]
    answer: Optional[str]
    citations: List[Any]
    stage: str


def controller(query: str) -> ControllerResult:

    with tracer.start_as_current_span("main_pipeline") as span:
        span.set_attribute("query", query)

        router_result = router.run(query)

        if router_result.system_failure:
            span.set_attribute("pipeline.stage", "router")
            span.set_attribute("pipeline.status", "failed")
            return ControllerResult(
                system_failure="router_failed",
                answer=None,
                citations=[],
                stage="router"
            )

        retrieval_result = run_retrieval_pipeline(router_result)

        if retrieval_result.system_failure:
            span.set_attribute("pipeline.stage", "retrieval")
            span.set_attribute("pipeline.status", "failed")
            return ControllerResult(
                system_failure="retrieval_failed",
                answer=None,
                citations=[],
                stage="retrieval"
            )

        generation_result = run_generation_pipeline(retrieval_result)

        if generation_result.system_failure:
            span.set_attribute("pipeline.stage", "generation")
            span.set_attribute("pipeline.status", "failed")
            return ControllerResult(
                system_failure="generation_failed",
                answer=None,
                citations=[],
                stage="generation"
            )

        span.set_attribute("pipeline.status", "success")
        return generation_result
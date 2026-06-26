"""
Generation orchestrator module.
Responsible for managing the answer generation pipeline, including context building,
answer synthesis, citation resolution, and validation.
"""

import logging
from opentelemetry import trace

from contracts.orchestration_contracts import (
    RetrievalLayerOutput, 
    GenerationLayerOutput,
    ExceptionInfo
)

from contracts.generation_contracts import (
    GenerationStatus, 
    GenerationValidationResult, 
    ValidationSignals
)

from generation_layer.context_builder import make_generation_context
from generation_layer.generation import AnswerGeneration
from generation_layer.generation_validation import validate_answer

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class GenerationOrchestrator:
    def __init__(self, model):
        self.generator = AnswerGeneration(model)

    def run(self, input: RetrievalLayerOutput, config: dict = None, chat_history: list = None) -> GenerationLayerOutput:
        """
        Executes the generation pipeline using the retrieved context.
        Returns the generated answer, resolved citations, and validation results,
        handling and logging any infrastructure failures.
        """

        with tracer.start_as_current_span("generation_pipeline") as span:

            span.set_attribute("generation.query", input.plan.original_query)

            try:
                generation_context = make_generation_context(input, chat_history=chat_history)
                span.set_attribute("generation.context_length", len(generation_context.context))

                answer_text = self.generator.generate_answer(generation_context, config=config)
                span.set_attribute("generation.answer_length", len(answer_text))
                
                citations = self.generator.resolve_citations(generation_context, answer_text)
                span.set_attribute("generation.citation_count", len(citations))

                validation_result = validate_answer(answer_text, generation_context.original_query)
                span.set_attribute("generation.validation_status", validation_result.status.value)

                return GenerationLayerOutput(
                    answer=answer_text,
                    citations=citations,
                    validation_result=validation_result,
                    system_failure=None,
                )

            except Exception as e:

                span.record_exception(e)
                span.set_attribute("generation.status", "error")
                logger.exception("Infrastructure failure in generation pipeline")

                return GenerationLayerOutput(
                    answer="",
                    citations=[],
                    validation_result=GenerationValidationResult(
                        status=GenerationStatus.EXCEPTION,
                        score=0.0,
                        signals=ValidationSignals(
                            has_citations=False,
                            citation_count=0,
                            answer_length=0,
                            coverage_score=0.0,
                            has_refusal_pattern=False,
                        ),
                        failure_reason="generation_infrastructure_failure",
                    ),
                    system_failure=ExceptionInfo(
                        exception_type=type(e).__name__,
                        message=str(e),
                    ),
                ) 
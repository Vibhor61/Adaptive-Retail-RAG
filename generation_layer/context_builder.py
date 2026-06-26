"""
Provides utilities to build the generation context from retrieval layer outputs.
Formats chunks of retrieved content alongside metadata like source and ASIN.
Constructs structured contexts suitable for downstream generation processing.
Uses OpenTelemetry for tracing and monitoring the context assembly process.
"""

from opentelemetry import trace

from contracts.orchestration_contracts import (
    RetrievalLayerOutput
)

from contracts.retrieval_contracts import (
    RetrievalQualityStatus,
    RetrievalResult
)

from contracts.generation_contracts import (
    GenerationContext
)

tracer = trace.get_tracer(__name__)

def format_context_chunk(citation_key: str,item: RetrievalResult) -> str:
    """
    Formats a single retrieval result item into a context string block.
    Includes the citation key, source origin, ASIN, and retrieved text.
    Returns a cleaned and formatted text chunk representing the item.
    """
    source = item.source or "unknown"

    return f"""
        [{citation_key}]
        SOURCE: {source.upper()}
        ASIN: {item.asin or "UNKNOWN"}

        CONTENT:
        {item.text}
        """.strip()


def make_generation_context(retrieval_output: RetrievalLayerOutput, chat_history: list = None) -> GenerationContext:
    """
    Constructs a comprehensive generation context from retrieval layer outputs.
    Iterates through valid evaluation bundles to assemble formatted context chunks.
    Returns a GenerationContext object containing the aggregated context and citations.
    """
    with tracer.start_as_current_span("generation.make_generation_context") as span:

        context_parts: list[str] = []
        citation_lookup: dict[str, RetrievalResult] = {}
        citation_counter = 1

        span.set_attribute("context.bundle_count", len(retrieval_output.evaluation_bundles))

        for evaluation_bundle in retrieval_output.evaluation_bundles:

            if evaluation_bundle.quality_status in (
                RetrievalQualityStatus.EMPTY, RetrievalQualityStatus.FAILED,
            ):
                continue

            bundle = evaluation_bundle.bundle
            if not bundle or not bundle.items:
                continue

            context_parts.append(f"\n=== {bundle.retrieval_type.upper()} ===\n")

            for item in bundle.items:

                citation_key = f"CTX_{citation_counter}"
                citation_lookup[citation_key] = item
                citation_counter += 1

                context_parts.append(
                    format_context_chunk(citation_key=citation_key, item=item)
                )

        context_str = "\n\n".join(context_parts)
        span.set_attribute("context.citation_count", len(citation_lookup))
        span.set_attribute("context.length", len(context_str))

        return GenerationContext(
            original_query=retrieval_output.plan.original_query,
            intent_type=retrieval_output.plan.intent_type.value,
            context=context_str,
            citation_lookup=citation_lookup,
            chat_history=chat_history or []
        )
    

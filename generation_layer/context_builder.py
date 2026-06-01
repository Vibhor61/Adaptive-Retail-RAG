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

    if item.source == "sparse_product":

        title = item.metadata.get("title", "")
        brand = item.metadata.get("brand", "")
        category = item.metadata.get("category", "")
        price = item.metadata.get("price", "")

        return f"""
            [{citation_key}]

            PRODUCT FACT

            ASIN: {item.asin}

            Title: {title}
            Brand: {brand}
            Category: {category}
            Price: {price}
            """.strip()

    return f"""
        [{citation_key}]

        CUSTOMER REVIEW

        ASIN: {item.asin}

        Review:
        {item.text}
        """.strip()


def make_generation_context(retrieval_output: RetrievalLayerOutput) -> GenerationContext:

    with tracer.start_as_current_span("make_generation_context"):

        context_parts: list[str] = []
        citation_lookup: dict[str, RetrievalResult] = {}
        citation_counter = 1

        for evaluation_bundle in retrieval_output.evaluation_bundles:

            if evaluation_bundle.quality_status in (
                RetrievalQualityStatus.EMPTY,RetrievalQualityStatus.FAILED,
            ):
                continue

            bundle = evaluation_bundle.bundle

            context_parts.append( f"\n=== {bundle.retrieval_type.upper()} ===\n")

            for item in bundle.items:

                citation_key = f"CTX_{citation_counter}"
                citation_lookup[citation_key] = item
                citation_counter += 1

                context_parts.append(
                    format_context_chunk(citation_key=citation_key, item=item)
                )

        return GenerationContext(
            original_query=retrieval_output.plan.original_query,
            intent_type=retrieval_output.plan.intent_type,
            context="\n\n".join(context_parts),
            citation_lookup=citation_lookup
        )
    

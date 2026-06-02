import logging
import re

from langchain_ollama import OllamaLLM
from opentelemetry import trace

from utils import safe_llm_call

from contracts.generation_contracts import(
    GenerationContext,
    GeneratedCitation,
)

from generation_layer.prompts import(
    build_lookup_prompt,
    build_comparison_prompt,
    build_recommendation_prompt
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
llm = OllamaLLM(model="qwen2.5:7b", temperature=0)

CTX_PATTERN = re.compile(f"\[CTX_(\d+)\]")


def select_prompt(context: GenerationContext) -> str:
    intent = context.intent_type

    if intent == "lookup":
        return build_lookup_prompt(context.context, context.original_query)
    
    elif intent == "comparison":
        return build_comparison_prompt(context.context, context.original_query)
    
    elif intent == "recommendation":
        return build_recommendation_prompt(context.context, context.original_query)
    
    else:
        raise ValueError(f"Unsupported intent type: {intent} — will be implemented later in adaptive routing")
    


def resolve_citations(context: GenerationContext, answer: str):

    with tracer.start_as_current_span("resolve_citations") as span:
        citations = []
        used_ctx_ids = set(CTX_PATTERN.findall(answer))
        span.set_attribute("citations.referenced_count", len(used_ctx_ids))
 
        for ctx_num in sorted(used_ctx_ids, key=int):
            ctx_key = f"CTX_{ctx_num}"
            result = context.citation_lookup.get(ctx_key)
            if result is None:
                continue
            citations.append(
                GeneratedCitation(
                    citation_id=ctx_key,
                    asin=result.asin,
                    review_id=result.review_id,
                    evidence_text=result.text,
                    retrieval_type=result.source,
                )
            )
 
        span.set_attribute("citations.resolved_count", len(citations))
        return citations


def generate_answer(context: GenerationContext) -> str:

    with tracer.start_as_current_span("answer_generation") as span:

        span.set_attribute("generation.query", context.original_query)
        span.set_attribute("generation.context_keys", len(context.citation_lookup))

        prompt = select_prompt(context)

        result = safe_llm_call(llm, prompt, mode="raw")

        answer_text = result["raw"].strip()

        span.set_attribute("answer.length", len(answer_text))

        return answer_text
"""
This module handles the core logic for generating natural language answers.
It leverages different prompting strategies based on user intent and uses OpenTelemetry for tracing.
The module interacts with LLM models to produce responses while also
maintaining and resolving context citations to ensure traceability.
"""

import logging

from opentelemetry import trace

from config.settings import settings
from utility_functions.llm_utils import extract_citation_ids

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


class AnswerGeneration:
    def __init__(self, model):
        self.model = model

    def select_prompt(self, context: GenerationContext) -> str:
        """
        Determines and constructs the appropriate prompt template based on the intent type.
        It accepts a GenerationContext object and returns a formatted prompt string tailored
        for 'lookup', 'comparison', or 'recommendation' intents.
        """
        intent = context.intent_type

        if intent == "lookup":
            return build_lookup_prompt(context.context, context.original_query, chat_history=context.chat_history)
        
        elif intent == "comparison":
            return build_comparison_prompt(context.context, context.original_query, chat_history=context.chat_history)
        
        elif intent == "recommendation":
            return build_recommendation_prompt(context.context, context.original_query, chat_history=context.chat_history)
        
        else:
            raise ValueError(f"Unsupported intent type: {intent} — will be implemented later in adaptive routing")
        


    def resolve_citations(self, context: GenerationContext, answer: str):
        """
        Extracts citation markers from the generated answer and maps them to their respective context entries.
        Returns a list of GeneratedCitation objects containing details like ASIN, review ID, and evidence text,
        which grounds the generated response in the provided context.
        """

        with tracer.start_as_current_span("resolve_citations") as span:
            citations = []
            used_ctx_ids = extract_citation_ids(answer)
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


    def generate_answer(self, context: GenerationContext, config: dict = None) -> str:
        """
        Generates an answer using the chosen prompt and the underlying LLM model.
        Tracks the generation process with OpenTelemetry, invoking the model safely
        and returning the raw generated answer string.
        """

        with tracer.start_as_current_span("answer_generation") as span:

            span.set_attribute("generation.query", context.original_query)
            span.set_attribute("generation.context_keys", len(context.citation_lookup))
            span.set_attribute("generation.context_size", len(context.context))
            span.set_attribute("generation.intent", context.intent_type)
            
            prompt = self.select_prompt(context)

            cfg = config or {}
            cfg["tags"] = cfg.get("tags", []) + ["generation"]

            # Notify any StreamingCallback handlers that generation is starting.
            # Uses duck typing — no import of main.py needed.
            _handlers = cfg.get("callbacks") or []
            if not isinstance(_handlers, list):
                _handlers = getattr(_handlers, "handlers", [])
            for _h in _handlers:
                if hasattr(_h, "start_generation"):
                    _h.start_generation()

            chunks = []
            try:
                for chunk in self.model.stream(prompt, config=cfg):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        chunks.append(piece)
            finally:
                for _h in _handlers:
                    if hasattr(_h, "stop_generation"):
                        _h.stop_generation()

            answer_text = "".join(chunks).strip()

            span.set_attribute("answer.length", len(answer_text))

            return answer_text
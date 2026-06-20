import json
import logging
import re

from opentelemetry import trace

from contracts.router_contracts import Intent, EntityStructure
from utils import safe_llm_call

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

INTENT_TOKEN_MAP = {
    "lookup": Intent.LOOKUP,
    "comparison": Intent.COMPARISON,
    "recommend": Intent.RECOMMENDATION,
    "recommendation": Intent.RECOMMENDATION,
    "unknown": Intent.UNKNOWN,
}

LLM_PROMPT = """You are an intent and entity classifier for an electronics retail search system.

Return a JSON object with exactly these two fields:
- intent: one of [lookup, comparison, recommendation, unknown]
- entities: list of specific product names or brands mentioned

DEFINITIONS

1. comparison
Use when the user is asking to compare, aggregate, summarize, find commonalities, find differences, or analyze multiple products.
This includes BOTH:
A) Explicit comparison
- compare X vs Y
- X or Y
- difference between X and Y
- which is better

B) Implicit/category comparison
- common features across X products
- similarities among X products
- differences across X products
- what do X products have in common
- how do reviews vary across X products
- summarize characteristics of X products

2. recommendation
Use when the user wants suggestions or options without a specific product in mind.
Signals: best, recommend, suggest, alternatives, under budget, for gaming, for travel.
entities is always empty.

3. lookup
Use when the user asks about one or more specific products, models, or brands.
Includes: specs, price, review, compatibility, features, details.
entities has one or more items.

4. unknown
Use when the query is not about discovering or evaluating products.
Includes: order tracking, returns, refunds, greetings, general knowledge.
entities is always empty.

ENTITY EXTRACTION RULES

- Extract the FULL product identifier including its device context as a single string.
- Device context (e.g. "for iPhone 6", "for Samsung Galaxy Note 4") is part of the product identity, NOT a separate entity.
- Never split a product and its target device into two separate entities.
- Never extract a bare device name (e.g. "iPhone", "Samsung Galaxy") as a standalone entity unless the query is specifically about that device itself.
- For accessory queries, merge the accessory name and device name into one entity string.

EXAMPLES

Query: reviews for BrandX case for iPhone 6
Output: {{"intent": "lookup", "entities": ["BrandX case iPhone 6"]}}

Query: BrandA charger vs BrandB charger for MacBook Pro
Output: {{"intent": "comparison", "entities": ["BrandA charger MacBook Pro", "BrandB charger MacBook Pro"]}}

Query: specs of ModelX screen protector for LG G5
Output: {{"intent": "lookup", "entities": ["ModelX screen protector LG G5"]}}

Query: OtterBox Defender vs Speck CandyShell
Output: {{"intent": "comparison", "entities": ["OtterBox Defender", "Speck CandyShell"]}}

Query: What features are common across Microsoft Lumia cases?
Output: {{"intent": "comparison", "entities": ["Microsoft Lumia cases"]}}

Query: What do OtterBox phone cases generally have in common?
Output: {{"intent": "comparison", "entities": ["OtterBox phone cases"]}}

Query: best wireless earbuds under 50 dollars
Output: {{"intent": "recommendation", "entities": []}}

Query: best phone case for Samsung Galaxy S6
Output: {{"intent": "recommendation", "entities": []}}

Query: is BrandZ battery good for HTC One
Output: {{"intent": "lookup", "entities": ["BrandZ battery HTC One"]}}

Query: BrandA earbuds and BrandB headphones reviews
Output: {{"intent": "lookup", "entities": ["BrandA earbuds", "BrandB headphones"]}}

Query: track my order
Output: {{"intent": "unknown", "entities": []}}

Query: {query}
Output:"""


class IntentClassifier:
    def __init__(self, llm):
        self.llm = llm

    def entity_structure(self, intent, entities: list[str]) -> EntityStructure:
        
        if intent == Intent.COMPARISON:
            if len(entities) >= 2:
                return EntityStructure.MULTI_EXPLICIT

            return EntityStructure.MULTI_IMPLICIT
    
        if len(entities) >= 2:
            return EntityStructure.MULTI_EXPLICIT

        if len(entities) == 1:
            return EntityStructure.SINGLE

        return EntityStructure.MULTI_IMPLICIT


    def classify_intent(self, query: str) -> tuple[Intent, list[str], EntityStructure]:
        with tracer.start_as_current_span("router.classify_intent") as span:
            span.set_attribute("query", query)
            try:
                prompt = LLM_PROMPT.format(query=query)
                parsed = safe_llm_call(self.llm, prompt, mode="json")

                span.set_attribute("llm.parsed_output", json.dumps(parsed))
                intent_str = parsed.get("intent", "unknown").lower()
                entities = parsed.get("entities", [])

                intent = INTENT_TOKEN_MAP.get(intent_str, Intent.UNKNOWN)
            
                if intent in (Intent.RECOMMENDATION, Intent.UNKNOWN):
                    entities = []
                    entity_struct = EntityStructure.NONE
                else:
                    entity_struct = self.entity_structure(intent, entities)

                span.set_attribute("llm.recognized", intent_str in INTENT_TOKEN_MAP)
                span.set_attribute("intent.value", intent.value)
                span.set_attribute("entities", str(entities))
                span.set_attribute("entity_structure", entity_struct.value)

                return intent, entities, entity_struct

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("intent.value", Intent.UNKNOWN.value)
                span.set_attribute("llm.failed", True)
                logger.warning(
                    "LLM intent classification failed for query=%r: %s", query, e
                )
                return Intent.UNKNOWN, [], EntityStructure.NONE   
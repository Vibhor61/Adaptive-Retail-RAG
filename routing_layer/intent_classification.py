"""
Provides functionality for determining user intent and extracting entities from queries.
Uses an LLM to classify intent into lookup, comparison, recommendation, or unknown,
and extracts product entities to determine the query's structural complexity.
"""
import json
import logging

from opentelemetry import trace

from contracts.router_contracts import Intent, EntityStructure
from utility_functions.llm_utils import safe_llm_call

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

Output MUST be a JSON object, not python code. Do NOT output python code blocks, markdown blocks, or any explanation. Return raw JSON ONLY.

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
Use when the user wants suggestions, options, or is searching for products based on descriptive features, aesthetics, or categories without a specific known product in mind.
Signals: find, looking for, best, recommend, suggest, alternatives, under budget, for gaming, for travel.
entities is always empty.

3. lookup
Use ONLY when the user asks about one or more specific, named products, exact models, or specific brands. If the user is just describing what a product looks like or does, it is a recommendation, not a lookup.
Includes: specs, price, review, compatibility, features, details of a KNOWN item.
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

Query: Find playful character cases for a Samsung Galaxy S5 with a cute black design.
Output: {{"intent": "recommendation", "entities": []}}

Query: specs of ModelX screen protector for LG G5
Output: {{"intent": "lookup", "entities": ["ModelX screen protector LG G5"]}}

Query: OtterBox Defender vs Speck CandyShell
Output: {{"intent": "comparison", "entities": ["OtterBox Defender", "Speck CandyShell"]}}

Query: What features are common across Microsoft Lumia cases?
Output: {{"intent": "comparison", "entities": ["Microsoft Lumia cases"]}}

Query: best wireless earbuds under 50 dollars
Output: {{"intent": "recommendation", "entities": []}}

Query: is BrandZ battery good for HTC One
Output: {{"intent": "lookup", "entities": ["BrandZ battery HTC One"]}}

Query: track my order
Output: {{"intent": "unknown", "entities": []}}

Query: {query}
Output:"""


class IntentClassifier:
    def __init__(self, llm):
        self.llm = llm

    def entity_structure(self, intent, entities: list[str]) -> EntityStructure:
        """
        Determines the entity structure based on the classified intent and extracted entities.
        Classifies into SINGLE, MULTI_EXPLICIT, or MULTI_IMPLICIT based on entity counts.
        """
        
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
        """
        Uses an LLM to classify user intent and extract relevant product entities.
        Returns the resolved intent, the list of extracted entities, and the entity structure.
        """
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
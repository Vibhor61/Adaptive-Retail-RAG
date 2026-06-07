import json
import logging

from opentelemetry import trace

from contracts.router_contracts import Intent, EntityStructure

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
Use when the user is choosing between TWO or more specific products or brands.
Signals: vs, versus, compare, difference between, which is better, X or Y.
entities must have 2 or more items.

2. recommendation
Use when the user wants suggestions or options without a specific product in mind.
Signals: best, recommend, suggest, alternatives, under budget, for gaming, for travel.
entities is always empty.

3. lookup
Use when the user asks about ONE specific product, model, or brand.
Includes: specs, price, review, compatibility, features, details.
entities has exactly 1 item.

4. unknown
Use when the query is not about discovering or evaluating products.
Includes: order tracking, returns, refunds, greetings, general knowledge.
entities is always empty.

EXAMPLES
Query: AirPods Pro vs Sony WH-1000XM5 for gym
Output: {{"intent": "comparison", "entities": ["AirPods Pro", "Sony WH-1000XM5"]}}

Query: best wireless earbuds under 5000
Output: {{"intent": "recommendation", "entities": []}}

Query: is Samsung S24 good for photography
Output: {{"intent": "lookup", "entities": ["Samsung S24"]}}

Query: Ghostek Blitz Note 4 case review
Output: {{"intent": "lookup", "entities": ["Ghostek Blitz Note 4 case"]}}

Query: Amzer Crusta vs Ghostek Blitz which is better for Note 4
Output: {{"intent": "comparison", "entities": ["Amzer Crusta", "Ghostek Blitz"]}}

Query: track my order
Output: {{"intent": "unknown", "entities": []}}

Query: {query}
Output:"""


class IntentClassifier:
    def __init__(self, llm):
        self.llm = llm

    def entity_structure(self, entities: list[str]) -> EntityStructure:
        if len(entities) >= 2:
            return EntityStructure.MULTI_EXPLICIT

        if len(entities) == 1:
            return EntityStructure.SINGLE

        return EntityStructure.MULTI_IMPLICIT


    def classify_intent(self, query: str) -> tuple[Intent, list[str], EntityStructure]:
        with tracer.start_as_current_span("router.classify_intent") as span:
            span.set_attribute("query", query)
            try:
                response = self.llm.invoke(LLM_PROMPT.format(query=query))
                raw = response.content.strip()

                span.set_attribute("llm.raw_output", raw)

                parsed = json.loads(raw)
                intent_str = parsed.get("intent", "unknown").lower()
                entities = parsed.get("entities", [])

                intent = INTENT_TOKEN_MAP.get(intent_str, Intent.UNKNOWN)
            
                if intent in (Intent.RECOMMENDATION, Intent.UNKNOWN):
                    entities = []
                    entity_struct = EntityStructure.NONE
                else:
                    entity_struct = self.entity_structure(entities)

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
            
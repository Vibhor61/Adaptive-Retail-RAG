import logging
import re

from opentelemetry import trace

from contracts.router_contracts import EvidenceType, Intent

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

EVIDENCE_TOKEN_MAP = {
    "factual": EvidenceType.FACTUAL,
    "experiential": EvidenceType.EXPERIENTIAL,
    "mixed": EvidenceType.MIXED,
}

LLM_PROMPT = """You are an expert evidence type classifier for an electronics retail search system.
Your job is to determine what kind of data is required to answer the user's query.

Classify the user query into exactly one label: [factual, experiential, mixed]
Return ONLY one lowercase word from the options. Do not include any punctuation or explanations.

--------------------------------
DEFINITIONS & STRICT TIE-BREAKING RULES

1. factual
- Use when the user wants hard, objective, binary, or numeric product specifications.
- Signals: specs, price, dimensions, capacity, compatibility, storage options, RAM variants, processor model name.
- RULE: Even if a query mentions a complex configuration (e.g., "compatibility of Anker with MacBook Pro"), if it can be answered with a 100% objective "Yes/No" or a spec sheet lookup, it is strictly FACTUAL.

2. experiential
- Use when the user wants subjective, qualitative, or real-world usage feedback.
- Signals: comfort, durability, overheating, battery drain, lag, smoothness, reliability, user complaints, reviews, "is it worth it".
- RULE: Adjectives and real-world conditions OVERRIDE hardware nouns. If a query asks about a subjective state of a specific component (e.g., "does MacBook Air overheat during coding"), it is strictly EXPERIENTIAL because it requires human field testing.

3. mixed
- Use ONLY when the user explicitly combines two distinct intents: asking for hard specifications AND asking for subjective user reviews in the same query.
- Do not use 'mixed' just because a query mentions a specific product name alongside an experiential question. It must ask for BOTH data types.

--------------------------------
EXAMPLES

Query: "processor used in Google Pixel 8"
Answer: factual

Query: "storage variants of iPhone 13"
Answer: factual

Query: "compatibility of Anker 737 charger with MacBook Pro"
Answer: factual

Query: "does MacBook Air overheat during coding"
Answer: experiential

Query: "how reliable is Samsung S24 camera in low light"
Answer: experiential

Query: "is gaming performance smooth on iPhone 14"
Answer: experiential

Query: "compare real world performance and specs of Snapdragon 8 Gen 3 phones"
Answer: mixed

Query: "should I buy AirPods Pro or Sony WF-1000XM5 for gym and sound quality"
Answer: mixed

--------------------------------
Query: {query}
Answer:"""


class EvidenceClassifier:
    def __init__(self, llm):
        self.llm = llm

    def llm_evidence_classification(self, query: str) -> EvidenceType:
        with tracer.start_as_current_span("router.evidence_llm") as span:
            span.set_attribute("query", query)
            try:
                response = self.llm.invoke(LLM_PROMPT.format(query=query))

                raw = response.content.strip().lower()

                span.set_attribute("llm.raw_output", raw)
                span.set_attribute("llm.recognized", raw in EVIDENCE_TOKEN_MAP)

                match = re.search(
                    r"\b(factual|experiential|mixed)\b",
                    raw
                )

                if match:
                    token = match.group(1)
                    evidence = EVIDENCE_TOKEN_MAP[token]
                else:
                    evidence = EvidenceType.MIXED
                span.set_attribute("evidence_type", evidence.value)
                return evidence

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("evidence_type", EvidenceType.MIXED.value)
                span.set_attribute("llm.failed", True)
                logger.warning(
                    "LLM evidence classification failed for query=%r: %s", query, e
                )
                return EvidenceType.MIXED

    def derive_evidence_type(self, intent: Intent, query: str) -> EvidenceType:
        with tracer.start_as_current_span("router.evidence_derive") as span:
            span.set_attribute("intent", intent.value)

            if intent == Intent.RECOMMENDATION:
                span.set_attribute("evidence_type", EvidenceType.MIXED.value)
                span.set_attribute("path", "intent_shortcircuit")
                return EvidenceType.MIXED

            evidence = self.llm_evidence_classification(query)

            span.set_attribute("evidence_type", evidence.value)
            return evidence
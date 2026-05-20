import re
import enum
import json
from dataclasses import dataclass
from typing import Set, Optional
from opentelemetry import trace
from langchain_ollama import OllamaLLM
from utils import safe_llm_call

tracer = trace.get_tracer(__name__)


class QueryType(enum.Enum):
    PRODUCT_LOOKUP = "product_lookup"
    REVIEW_QUERY = "review_query"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    query_type: QueryType
    confidence: float
    reason: str
    llm_used: bool


SPARSE_KEYWORDS = {"price", "cost", "brand", "category", "cheap", "expensive", "rate"}
DENSE_KEYWORDS = {"best", "good", "bad", "worst", "review", "recommend", "comfortable", "quality", "worth", "experience"}
HYBRID_KEYWORDS = {"compare", "vs", "versus", "difference", "better", "between"}


router_llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0)


def preprocess(query: str):
    return set(re.findall(r"\w+", query.lower()))


def detect_query_type(tokens: Set[str]) -> QueryType:
    sparse_match = bool(tokens & SPARSE_KEYWORDS)
    dense_match = bool(tokens & DENSE_KEYWORDS)
    hybrid_match = bool(tokens & HYBRID_KEYWORDS)

    if hybrid_match:
        return QueryType.COMPARISON
    if sparse_match and dense_match:
        return QueryType.REVIEW_QUERY
    if dense_match:
        return QueryType.REVIEW_QUERY
    if sparse_match:
        return QueryType.PRODUCT_LOOKUP
    return QueryType.UNKNOWN


def llm_fallback(query: str) -> IntentResult:
    llm = router_llm

    with tracer.start_as_current_span("intent_llm_fallback") as span:
        span.set_attribute("intent.query", query)

        prompt = f"""
        Classify query into:
        product_lookup | review_query | comparison

        Return ONLY JSON:
        {{"query_type": "..."}}

        Query: {query}
        """

        try:
            parsed = safe_llm_call(llm, prompt, "json")

            span.set_attribute("intent.llm_fallback","success")
            span.set_attribute("llm_answer",parsed)

            qtype = QueryType(parsed["query_type"])

            return IntentResult(
                query_type=qtype,
                confidence=0.85,
                reason="llm_classification",
                llm_used=True
            )

        except Exception as e:
            span.set_attribute("intent.error", str(e))

            return IntentResult(
                query_type=QueryType.REVIEW_QUERY,
                confidence=0.5,
                reason="llm_fallback_failed",
                llm_used=True
            )


def analyze_intent(query: str) -> IntentResult:
    tokens = preprocess(query)
    query_type = detect_query_type(tokens)

    if query_type == QueryType.UNKNOWN:
        return llm_fallback(query)

    return IntentResult(
        query_type=query_type,
        confidence=0.75,
        reason="keyword_based_intent",
        llm_used=False
    )
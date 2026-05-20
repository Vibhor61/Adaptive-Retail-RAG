import re
import enum
import json
from dataclasses import dataclass
from typing import Set
from opentelemetry import trace
from langchain_ollama import OllamaLLM

tracer = trace.get_tracer(__name__)

class QueryType(enum.Enum):
    PRODUCT_LOOKUP = "product_lookup"
    REVIEW_QUERY = "review_query"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    query_type: QueryType
    tokens: Set[str]
    is_question: bool
    has_numbers: bool
    length: int


SPARSE_KEYWORDS = {
    "price", "cost", "brand", "category", "cheap", "expensive", "rate"
}

DENSE_KEYWORDS = {
    "best", "good", "bad", "worst", "review",
    "recommend", "comfortable", "quality", "worth", "experience"
}

HYBRID_KEYWORDS = {
    "compare", "vs", "versus", "difference", "better", "between"
}

router_llm = OllamaLLM("qwen2.5:1.5b")

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


def is_noise(query: str, tokens: Set[str]) -> bool:

    cleaned = re.sub(r"\s+", "", query)

    if len(cleaned) <= 2:
        return True

    if re.fullmatch(r"[\W\d_]+", cleaned):
        return True

    if len(tokens) == 0:
        return True

    return False



def llm_fallback(query:str) -> QueryType:

    with tracer.start_as_current_span("router_llm_fallback") as span:

        span.set_attribute("router.query", query)

        prompt = f"""
            You are an e-commerce query intent classifier.
            Classify the query into one of:
            - product_lookup
            - review_query
            - comparison

            Return ONLY valid JSON:
            {{
                "query_type": "<type>"
            }}

            Query:"{query}"
            
        """

        try:
            raw = router_llm.invoke(prompt)
            text = raw if isinstance(raw, str) else raw.content
            parsed = json.loads(text.strip())
            query_type = parsed["query_type"]

            span.set_attribute("llm.parse.status", "success")
            return QueryType(query_type)

        except Exception as e:

            span.set_attribute("llm.parse.status", "failure")
            span.set_attribute("router.failure_reason", str(e))

            return QueryType.REVIEW_QUERY


def preprocess(query: str):
    query = query.lower()
    query = re.sub(r'[^\w\s]', ' ', query)
    return query.split()


def analyze_intent(query: str) -> IntentResult:

    tokens = preprocess(query)

    query_type = detect_query_type(tokens)

    result = IntentResult(
        query_type=query_type,
        tokens=tokens,
        reason="deterministic intent extraction",
        is_question="?" in query,
        has_numbers=bool(re.search(r"\d", query)),
        length=len(tokens),
    )

    return result
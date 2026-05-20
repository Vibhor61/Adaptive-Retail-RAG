import re
from dataclasses import dataclass
import enum
from typing import Optional
import re
from opentelemetry import trace

from intent import (
    analyze_intent,
    llm_fallback,
    QueryType,
    IntentResult
)

from validity import (
    is_query_valid,
    QueryValidity,
    ValidationResult
)

from resolver import DBEntityLoader

tracer = trace.get_tracer(__name__)

class RouteType:
    RETRIEVE = "retrieve"
    REWRITE = "rewrite"
    CLARIFY = "clarify"                 


class RetrievalType(enum.Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    NONE = "none"


@dataclass
class RouterResult:

    validation_result : ValidationResult
    
    intent_result : 
    retrieval_type: str
    query_type: QueryType

    entity_match: Optional[str]
    entity_score: float
    entity_strategy: Optional[str]

    llm_used: bool


def preprocess_query(query:str) -> list[str]:
    query = query.lower()
    query = re.sub(r'[^\w\s]', ' ', query)
    return query.split()


class QueryRouter:
    def __init__(self):
        self.resolver = DBEntityLoader()

    def map_retrieval_type(self, query_type: QueryType) -> RetrievalType:
        if query_type == QueryType.PRODUCT_LOOKUP:
            return RetrievalType.SPARSE

        if query_type == QueryType.REVIEW_QUERY:
            return RetrievalType.DENSE

        if query_type == QueryType.COMPARISON:
            return RetrievalType.HYBRID

        return RetrievalType.NONE
    
   def route(self, query: str) -> RouterResult:

        valid = is_query_valid(query)

        if not valid.validation_result.Query_validity == "invalid":
            return 

        # -------------------------
        # 2. INTENT
        # -------------------------
        intent = analyze_intent(query)

        entity, score, strategy = self.resolver.resolve(query)
   
        if entity and score >= 0.85:
            return RouteResult(
                route_type=RouteType.RETRIEVE,
                retrieval_type=RetrievalType.ENTITY,
                entity=entity,
                query_type=intent.query_type.value,
                confidence=score,
                reason=f"entity_match::{strategy}"
            )

        intent_conf = 0.7

        confidence = (
            0.4 * v.confidence +
            0.3 * intent_conf +
            0.3 * score
        )

        if confidence < 0.55:
            return RouteResult(
                route_type=RouteType.REWRITE,
                retrieval_type=None,
                entity=entity,
                query_type=intent.query_type.value,
                confidence=confidence,
                reason="low_confidence"
            )

        return RouteResult(
            route_type=RouteType.RETRIEVE,
            retrieval_type=rtype,
            entity=entity,
            query_type=intent.query_type.value,
            confidence=confidence,
            reason="intent_routing"
        )
        

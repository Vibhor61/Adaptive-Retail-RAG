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

class RouterAction(enum.Enum):
    CONTINUE = "continue"
    REWRITE = "rewrite"
    CLARIFY = "clarify"                   


class RetrievalType(enum.Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    NONE = "none"

@dataclass
class RouterResult:
    query_validity: QueryValidity
    retrieval_type: str
    query_type: QueryType

    next_action: RouterAction

    entity_match: Optional[str]
    entity_score: float
    entity_strategy: Optional[str]

    validation: ValidationResult
    intent: Optional[IntentResult]

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
    
    def route(self, query:str):
        

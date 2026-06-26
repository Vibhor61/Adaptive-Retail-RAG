"""
Data contracts for the generation phase of the RAG pipeline.
Defines Pydantic models and enums for generation contexts, statuses, and validation signals.
Ensures type safety and structured data handling during answer generation.
"""
import enum 

from pydantic import BaseModel
from typing import Optional 

from contracts.retrieval_contracts import RetrievalResult

class GenerationContext(BaseModel):
    original_query: str

    intent_type: str

    context: str

    citation_lookup: dict[str, RetrievalResult]

    chat_history: list = []


class GenerationStatus(enum.Enum): 
    PASSED = "passed" 

    EMPTY = "empty" 

    TOO_SHORT = "too_short" 

    REFUSAL = "refusal" 

    LOW_COVERAGE = "low_coverage" 

    ATTRIBUTION_ERROR = "attribution_error"

    EXCEPTION = "exception" 


class ValidationSignals(BaseModel): 
    has_citations: bool 

    citation_count: int

    answer_length: int 

    coverage_score: float

    has_refusal_pattern: bool


class GenerationValidationResult(BaseModel):
    status: GenerationStatus

    score: float

    signals: ValidationSignals

    failure_reason: Optional[str] = None


class GeneratedCitation(BaseModel):
    citation_id: str

    asin: str

    review_id: Optional[str]

    evidence_text: str

    retrieval_type: str

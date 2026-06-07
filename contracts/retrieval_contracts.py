import enum 

from typing import Optional
from pydantic import BaseModel, Field
from contracts.router_contracts import (
    Intent,
    EvidenceType,
    EntityStructure,
    RankedCandidate
)

"""
Data Model for Retrieval Input
"""
class RetrievalPlan(BaseModel):

    original_query: str

    intent_type: Intent

    evidence_type: EvidenceType

    entity_structure: EntityStructure

    grounded_entities: list[RankedCandidate]

    # constraints: QueryConstraints | None

    top_k: int = 5

    # retrieval_attempt: int = 0

    # rewritten_query: str | None = None

    # controller_actions: list[str] = Field(default_factory=list)


"""
Data Models for Retreival Execution
"""

class RetrievalExecutionStatus(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class RetrievalResult(BaseModel):
    source: str

    doc_id: str

    score: float

    rank: int

    text: str

    asin: Optional[str] = None

    review_id: Optional[str] = None

    metadata: dict = Field(default_factory=dict)


class RetrievalRawSignals(BaseModel):
    top_score: float

    avg_score: float

    score_distribution: list[float]


class RetrievalBundle(BaseModel):
    entity: Optional[str] = None  
    
    query: Optional[str] = None

    retrieval_type: str

    execution_status: RetrievalExecutionStatus

    items: list[RetrievalResult]

    raw_signals: Optional[RetrievalRawSignals] = None

    failure_reason: Optional[str] = None


"""
Data Models for Signals and Evaluation
"""

class RetrievalQualityStatus(enum.Enum):
    HEALTHY = "healthy"
    WEAK = "weak"
    EMPTY = "empty"
    FAILED = "failed"


class RetrievalEvaluationSignals(BaseModel):
    retrieval_type: str

    total_items: int

    unique_asins: int


"""
Data Model for Final Output
"""
class RetrievalEvaluationBundle(BaseModel):
    bundle: RetrievalBundle

    query: str | None = None

    quality_status: RetrievalQualityStatus

    signals: RetrievalEvaluationSignals

    anomaly_flags: list[str]
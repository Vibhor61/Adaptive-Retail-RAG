import enum 

from typing import Optional
from dataclasses import dataclass, field


"""
Data Models for Retreival Execution
"""

class RetrievalExecutionStatus(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class RetrievalResult:
    source: str

    doc_id: str

    score: float

    rank: int

    text: str

    asin: Optional[str] = None

    review_id: Optional[int] = None

    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRawSignals:
    top_score: float

    avg_score: float

    score_distribution: list[float]


@dataclass(frozen=True)
class RetrievalBundle:
    query: str

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


@dataclass(frozen=True)
class RetrievalEvaluationSignals:
    retrieval_type: str

    total_items: int

    top_score: float
    avg_score: float

    unique_asins: Optional[int]

    score_variance: float

    lexical_dense_overlap: Optional[int] = None



"""
Data Model for Final Output
"""
@dataclass(frozen=True)
class RetrievalEvaluationBundle:
    bundle: RetrievalBundle

    quality_status: RetrievalQualityStatus

    confidence: float

    signals: RetrievalEvaluationSignals

    anomaly_flags: list[str]
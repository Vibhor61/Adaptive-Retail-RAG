"""
Defines data models and contracts for the router layer.
Includes models for query validation, entity resolution, routing hypotheses,
intent parsing, and semantic guardrail validation.
"""
import enum

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Literal

"""
Data Models for Router Pre-Validation
"""

class ValidityStatus(enum.Enum):
    EXECUTABLE = "executable"
    SUSPICIOUS = "suspicious"
    DEGRADED = "degraded"


class ValidityFlags(enum.Enum):

    EMPTY_QUERY = "empty_query"

    EXCESSIVE_LENGTH = "excessive_length"

    HARD_LENGTH_REJECT = "hard_length_reject"

    CONTROL_CHARACTERS = "control_characters_detected"

    SYMBOL_SPAM = "symbol_spam_detected"

    CHARACTER_FLOOD = "character_flood_detected"

    HIGH_SYMBOL_RATIO = "high_symbol_ratio"


class ValidationResult(BaseModel):
    status: ValidityStatus
    
    normalized_query: str

    word_count: int

    anomaly_flags: list[ValidityFlags] = Field(default_factory=list)


"""
Data Models for Entity Resolution
"""
class MatchType(enum.Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    FTS_PRODUCT = "fts_product"
    NONE = "none"


class CandidateEntity(BaseModel):
    asin: Optional[str] = None

    title: Optional[str] = None

    brand: Optional[str] = None

    match_type: MatchType

    retrieval_score: float = 0.0


class RankedCandidate(BaseModel):
    asin: Optional[str]

    title: str

    brand: Optional[str]

    retrieval_score: float
    
    reranker_score: float

"""
Data Models for Router Hypotheses and Decisions
"""

class Intent(enum.Enum):
    LOOKUP = "lookup"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


class IntentResponse(BaseModel):
    intent: Intent
    entities: list[str]


class EntityStructure(enum.Enum):
    NONE = "none"
    SINGLE = "single"
    MULTI_EXPLICIT = "multi_explicit"
    MULTI_IMPLICIT = "multi_implicit"


class EvidenceType(enum.Enum):
    FACTUAL = "factual"
    EXPERIENTIAL = "experiential"
    MIXED = "mixed"


class QueryConstraints(BaseModel):
    raw_constraints: dict[str, Any] = Field(default_factory=dict)


class RouterResult(BaseModel):
   
    intent_type: Intent

    entities: List[RankedCandidate]

    entity_structure: EntityStructure

    evidence_type: EvidenceType

    constraints: QueryConstraints | None = None

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

"""
Data Models for Semantic Gaurdrails
"""

class SemanticAnomalyType(enum.Enum):

    INTENT_MISMATCH = "intent_mismatch"

    EVIDENCE_TYPE_MISMATCH = "evidence_type_mismatch"

    ENTITY_STRUCTURE_MISMATCH = "entity_structure_mismatch"

    INSUFFICIENT_ENTITY_SUPPORT = "insufficient_entity_support"

    LOW_SEMANTIC_ALIGNMENT = "low_semantic_alignment"

    AMBIGUOUS_ROUTING = "ambiguous_routing"


class SemanticValidationResult(BaseModel):

    semantic_valid: bool

    semantic_score: float = Field(
        ge=0.0,
        le=1.0
    )

    anomaly_signals: List[SemanticAnomalyType] = Field(
        default_factory=list
    )

    reasoning_confidence: float = Field(
        ge=0.0,
        le=1.0
    )

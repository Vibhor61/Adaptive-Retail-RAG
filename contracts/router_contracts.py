import enum
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Any, Optional

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
Data Models for Router Hypotheses and Decisions
"""

class Intent(enum.Enum):
    LOOKUP = "lookup"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


class EntityStructure(enum.Enum):
    NONE = "none"
    SINGLE = "single"
    MULTI_EXPLICIT = "multi_explicit"
    MULTI_IMPLICIT = "multi_implicit"


class EvidenceType(enum.Enum):
    FACTUAL = "factual"
    EXPERIENTIAL = "experiential"
    MIXED = "mixed"


class ExtractedEntity(BaseModel):
    text: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

class QueryConstraints(BaseModel):
    raw_constraints: dict[str, Any] = Field(default_factory=dict)


class RouterResult(BaseModel):
   
    intent_type: Intent

    entities: List[ExtractedEntity] = Field(
        default_factory=list
    )

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


"""
Data Models for Structured Guardrail Outputs
"""

class ViolationSeverity(enum.Enum):
    WARNING = "warning"
    ERROR = "error"


class StructuralViolation(BaseModel):
    field: str
    reason: str
    severity: ViolationSeverity


class StructuralGuardrailResult(BaseModel):
    passed: bool
    violations: List[StructuralViolation]

"""
Data Models for Entity Resolution
"""
class MatchType(enum.Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    FTS_PRODUCT = "fts_product"
    NONE = "none"


class GroundedEntity(BaseModel):
    match_type: MatchType
    raw_entity: str
    canonical_entity: Optional[str]
    score: float
from dataclasses import dataclass 
from typing import Optional 
import enum 


class GenerationStatus(enum.Enum): 
    PASSED = "passed" 

    EMPTY = "empty" 

    TOO_SHORT = "too_short" 

    REFUSAL = "refusal" 

    LOW_COVERAGE = "low_coverage" 

    ATTRIBUTION_ERROR = "attribution_error"

    EXCEPTION = "exception" 


@dataclass 
class ValidationSignals: 
    has_evidence: bool 

    answer_length: int 

    cited_evidence: bool 

    mentioned_entities: int 

    coverage_score: Optional[float] 


@dataclass 
class GenerationResult: 
    answer: str 

    model_used: str 

    status: GenerationStatus 

    score: float 

    signals: Optional[ValidationSignals] 

    failure_reason: Optional[str] 
    
    failure_details: Optional[str] = None
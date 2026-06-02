from typing import List, Optional
from pydantic import BaseModel

from contracts.router_contracts import (
    RouterResult,
    SemanticValidationResult,
    StructuralGuardrailResult,
    ValidationResult,
    GroundedEntity,
)

from contracts.retrieval_contracts import (
    RetrievalPlan,
    RetrievalEvaluationBundle,
)

from contracts.generation_contracts import (
    ValidationSignals,
    GenerationStatus,
    GeneratedCitation,
    GenerationValidationResult
)

"""
Data Models for Router Layer Output
"""

class ExceptionInfo(BaseModel):
    exception_type: str
    message: str


class RouterLayerOutput(BaseModel):

    normalized_query: str

    validity_result: ValidationResult

    router_output: RouterResult | None = None

    structural_result: StructuralGuardrailResult | None = None

    semantic_result: SemanticValidationResult | None = None

    grounded_entities: List[GroundedEntity]

    system_failure: ExceptionInfo | None = None

"""
Data Model for Retrieval Layer Output
"""

class RetrievalLayerOutput(BaseModel):

    plan: RetrievalPlan

    evaluation_bundles: list[RetrievalEvaluationBundle]

    system_failure: ExceptionInfo | None = None


"""
Data Model for Generation Layer Output
"""

class GenerationLayerOutput(BaseModel): 
    answer: str 

    # model_used: str 

    citations: list[GeneratedCitation]

    validation_result: GenerationValidationResult

    system_failure: ExceptionInfo | None = None
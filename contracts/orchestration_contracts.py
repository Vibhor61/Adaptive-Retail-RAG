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


class RouterLayerOutput(BaseModel):

    normalized_query: str

    validity_result: ValidationResult

    router_output: RouterResult

    structural_result: StructuralGuardrailResult

    semantic_result: SemanticValidationResult

    grounded_entities: List[GroundedEntity]


"""
Data Model for Retrieval Layer Output
"""

class RetrievalLayerOutput(BaseModel):

    plan: RetrievalPlan

    evaluation_bundles: list[RetrievalEvaluationBundle]


"""
Data Model for Generation Layer Output
"""

class GenerationLayerOutput(BaseModel): 
    answer: str 

    # model_used: str 

    citations: list[GeneratedCitation]

    validation_result: GenerationValidationResult

    failure_reason: Optional[str] = None
    
    failure_details: Optional[str] = None
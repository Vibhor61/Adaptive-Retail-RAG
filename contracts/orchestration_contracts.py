from typing import List
from pydantic import BaseModel

from contracts.router_contracts import (
    RouterResult,
    ValidationResult,
    RankedCandidate
)

from contracts.retrieval_contracts import (
    RetrievalPlan,
    RetrievalEvaluationBundle,
)

from contracts.generation_contracts import (
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

    grounded_entities: List[RankedCandidate]

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
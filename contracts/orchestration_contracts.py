from typing import List
from pydantic import BaseModel

# Data Models for Router orchestration output
from router_contracts import RouterOutput, SemanticValidationResult, StructuralGuardrailResult
from routing_layer.validity import ValidationResult
from routing_layer.entity_resolver import GroundedEntity

class RoutingResult(BaseModel):

    normalized_query: str

    validity_result: ValidationResult

    router_output: RouterOutput

    structural_result: StructuralGuardrailResult

    semantic_result: SemanticValidationResult

    grounded_entities: List[GroundedEntity]


"""
Data Model for Retrieval Layer Output
"""
from retrieval_contracts import RetrievalPlan, RetrievalEvaluationBundle

class RetrievalLayerOutput(BaseModel):

    plan: RetrievalPlan

    evaluation_bundles: list[RetrievalEvaluationBundle]
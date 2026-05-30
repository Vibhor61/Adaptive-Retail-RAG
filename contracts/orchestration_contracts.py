from typing import List
from pydantic import BaseModel

# Data Models for Router orchestration output
from contracts.router_contracts import RouterOutput, SemanticValidationResult, StructuralGuardrailResult
from routing_layer.validity import ValidationResult
from routing_layer.entity_resolver import GroundedEntity

class RoutingResult(BaseModel):

    original_query: str

    validity_result: ValidationResult

    router_output: RouterOutput

    structural_result: StructuralGuardrailResult

    semantic_result: SemanticValidationResult

    grounded_entities: List[GroundedEntity]

  
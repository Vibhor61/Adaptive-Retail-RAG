from dataclasses import dataclass
from typing import List, Dict, Any

from pydantic import BaseModel
from .router import (
    RouterOutput,
    QueryType,
    EntityStructure,
    AnswerType
)



class StructuralGuardrailResult(BaseModel):
    passed: bool

    consistency_score: float

    violations: List[str]


VALID_QUERY_TYPES = {
    "product_lookup",
    "product_attribute_query",
    "review_query",
    "comparison",
    "multi_product_reasoning",
    "recommendation",
    "unknown"
}

VALID_ENTITY_STRUCTURES = {
    "none",
    "single",
    "multi"
}

VALID_ANSWER_TYPES = {
    "factual",
    "experiential",
    "comparative",
    "recommendation",
    "unknown"
}


class StructuralGuardrails:


    @staticmethod
    def validate(router_output: RouterOutput) -> StructuralGuardrailResult:

        violations = []

        required_fields = [
            "query_type",
            "entities",
            "entity_structure",
            "answer_type",
            "anomaly_signals",
            "confidence"
        ]

        for field in required_fields:
            if field not in router_output:





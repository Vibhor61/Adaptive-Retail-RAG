import enum
from dataclasses import dataclass
from typing import List, Optional

from scripts.resolver.entity_resolver import MatchResult
from scripts.routing.router import IntentResult, QueryType

class RetrievalScope(enum.Enum):
    PRODUCT = "product"
    REVIEW = "review"
    BOTH = "both"
    NONE = "none"


class RetrievalMode(enum.Enum):
    METADATA = "metadata"
    REVIEW_SPARSE = "review_sparse" # review_fts
    REVIEW_DENSE = "review_dense"
    NONE = "none"


@dataclass
class RetrievalPlan:
    scope: RetrievalScope              
    modes: List[RetrievalMode]        

    resolved_entities: List[str]       # canonical only
    raw_entities: List[str]

    top_k: int

    router_confidence: float
    grounding_confidence: float

    proceed: bool 
    failure_reason: Optional[str]


CONFIDENCE_THRESHOLD = 0.5
RESOLVING_THRESHOLD = 0.6


class Planner():
    def __init__(self):
        pass
    
    def _failure_plan(
        self, 
        raw_entities: List[str], 
        resolved_entities: List[str], 
        router_confidence: float, 
        grounding_confidence: float, 
        reason: str
    ) -> RetrievalPlan:
        
        return RetrievalPlan(
            scope=RetrievalScope.NONE,
            modes=[RetrievalMode.NONE],

            resolved_entities=resolved_entities,
            raw_entities=raw_entities,

            top_k=0,

            router_confidence=router_confidence,
            grounding_confidence=grounding_confidence,

            proceed=False,
            failure_reason=reason
        )
    
    def plan(self, intent: IntentResult, resolved: List[MatchResult]) -> RetrievalPlan:
        
        canonical_entities = [ 
            c.resolved_entity  for c in resolved 
            if (
                c.resolved_entity is not None
                and c.score >= RESOLVING_THRESHOLD
            )
        ]
        
        raw_entities = [ 
            r.raw_entity for r in resolved
        ] # or can use intent result too 

        grounding_confidence = max(
            [r.score for r in resolved], default=0.0
        )

        router_confidence = intent.confidence

        if intent.query_type == QueryType.UNKNOWN :
            
            return self._failure_plan(
                raw_entities=raw_entities,
                resolved_entities=canonical_entities,
                router_confidence=router_confidence,
                grounding_confidence=grounding_confidence,
                reason="unknown query intent"
            )
        
        if intent.confidence < CONFIDENCE_THRESHOLD:

            return self._failure_plan(
                raw_entities=raw_entities,
                resolved_entities=canonical_entities,
                router_confidence=router_confidence,
                grounding_confidence=grounding_confidence,
                reason="low router confidence"
            )

        if intent.query_type == QueryType.PRODUCT_LOOKUP:

            if len(canonical_entities) == 0:
                return self._failure_plan(
                    raw_entities=raw_entities,
                    resolved_entities=[],
                    router_confidence=router_confidence,
                    grounding_confidence=grounding_confidence,
                    reason="product entity grounding failed"
                )

            return RetrievalPlan(
                scope=RetrievalScope.PRODUCT,
                modes=[RetrievalMode.METADATA],
                resolved_entities=canonical_entities,
                raw_entities=raw_entities,
                top_k=3,
                router_confidence=router_confidence,
                grounding_confidence=grounding_confidence,
                proceed=True,
                failure_reason=None
            )
        

        if intent.query_type == QueryType.PRODUCT_ATTRIBUTE_QUERY:

            if len(canonical_entities) == 0:
                return self._failure_plan(
                    raw_entities=raw_entities,
                    resolved_entities=[],
                    router_confidence=router_confidence,
                    grounding_confidence=grounding_confidence,
                    reason="product entity grounding failed"
                )

            return RetrievalPlan(
                scope=RetrievalScope.PRODUCT,
                modes=[RetrievalMode.METADATA],
                resolved_entities=canonical_entities,
                raw_entities=raw_entities,
                top_k=5,
                router_confidence=router_confidence,
                grounding_confidence=grounding_confidence,
                proceed=True,
                failure_reason=None
            )
        

        
        if intent.query_type == QueryType.REVIEW_QUERY:

            return RetrievalPlan(
                scope=RetrievalScope.REVIEW,
                modes=[
                    RetrievalMode.REVIEW_SPARSE,
                    RetrievalMode.REVIEW_DENSE
                ],
                resolved_entities=[],
                raw_entities=[],
                top_k=10,
                router_confidence=router_confidence,
                grounding_confidence=0.0,
                proceed=True,
                failure_reason=None
            )



        if intent.query_type == QueryType.HYBRID:

            if len(canonical_entities) == 0:
                return self._failure_plan(
                    raw_entities=raw_entities,
                    resolved_entities=[],
                    router_confidence=router_confidence,
                    grounding_confidence=grounding_confidence,
                    reason="hybrid entity grounding failed"
                )

            return RetrievalPlan(
                scope=RetrievalScope.BOTH,
                modes=[
                    RetrievalMode.METADATA,
                    RetrievalMode.REVIEW_SPARSE,
                    RetrievalMode.REVIEW_DENSE
                ],
                resolved_entities=canonical_entities,
                raw_entities=raw_entities,
                top_k=10,
                router_confidence=router_confidence,
                grounding_confidence=grounding_confidence,
                proceed=True,
                failure_reason=None
            )


        if intent.query_type == QueryType.COMPARISON:

            if len(canonical_entities) < 2:
                return self._failure_plan(
                    raw_entities=raw_entities,
                    resolved_entities=canonical_entities,
                    router_confidence=router_confidence,
                    grounding_confidence=grounding_confidence,
                    reason="insufficient grounded entities for comparison"
                )

            return RetrievalPlan(
                scope=RetrievalScope.BOTH,
                modes=[
                    RetrievalMode.METADATA,
                    RetrievalMode.REVIEW_SPARSE,
                    RetrievalMode.REVIEW_DENSE
                ],
                resolved_entities=canonical_entities,
                raw_entities=raw_entities,
                top_k=15,
                router_confidence=router_confidence,
                grounding_confidence=grounding_confidence,
                proceed=True,
                failure_reason=None
            )
        

        if intent.query_type == QueryType.MULTI_ENTITY_REVIEW:

            if len(canonical_entities) < 2:
                return self._failure_plan(
                    raw_entities=raw_entities,
                    resolved_entities=canonical_entities,
                    router_confidence=router_confidence,
                    grounding_confidence=grounding_confidence,
                    reason="multi entity review grounding failed"
                )

            return RetrievalPlan(
                scope=RetrievalScope.BOTH,
                modes=[
                    RetrievalMode.REVIEW_SPARSE,
                    RetrievalMode.REVIEW_DENSE
                ],

                resolved_entities=canonical_entities,
                raw_entities=raw_entities,

                top_k=15,

                router_confidence=router_confidence,
                grounding_confidence=grounding_confidence,

                proceed=True,
                failure_reason=None
            )
        

        return self._failure_plan(
            raw_entities=raw_entities,
            resolved_entities=canonical_entities,
            router_confidence=router_confidence,
            grounding_confidence=grounding_confidence,
            reason="unhandled planner state"
        )
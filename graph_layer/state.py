from typing import TypedDict, Optional, List, Literal
from contracts.router_contracts import RankedCandidate
from contracts.orchestration_contracts import (
    RouterLayerOutput,
    RetrievalLayerOutput,
    GenerationLayerOutput,
    StructuralGuardrailResult,
)


class ControlState(TypedDict):
    stage: Literal[
        "init",
        "rewrite",
        "route",
        "retrieve",
        "generate",
        "clarify",
        "end"
    ]

    should_clarify: bool

    failure_stage: Optional[Literal["router", "retrieval", "generation"]]
    failure_reason: Optional[str]


class QueryState(TypedDict):
    original_query: str
    rewritten_query: Optional[str]


class RouterState(TypedDict):
    output: Optional[RouterLayerOutput]
    guardrails: Optional[StructuralGuardrailResult]


class RetrievalState(TypedDict):
    output: Optional[RetrievalLayerOutput]
    guardrails: Optional[StructuralGuardrailResult]


class GenerationState(TypedDict):
    output: Optional[GenerationLayerOutput]


class ResponseState(TypedDict):
    answer: Optional[str]
    citations: List[dict]


class GraphState(TypedDict):

    query: QueryState
    chat_history: List[dict]

    control: ControlState

    router: RouterState
    retrieval: RetrievalState
    generation: GenerationState

    response: ResponseState

    grounded_entities: List[RankedCandidate]
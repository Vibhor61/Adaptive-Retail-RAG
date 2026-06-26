"""
This module is responsible for defining and constructing the overall LangGraph workflow.
It wires together nodes like rewriting, routing, retrieval, generation, and clarification.
The module sets up conditional edges based on guardrail checks to route execution appropriately.
"""

from langgraph.graph import StateGraph, END

from graph_layer.state import GraphState
from graph_layer.nodes import (
    make_rewrite_node,
    make_router_node,
    router_guardrail_node,
    post_router_node,
    retrieval_node,
    retrieval_guardrail_node,
    make_generation_node,
    clarification_node,
    route_after_router_guardrail,
    route_after_post_router,
    route_after_retrieval_guardrail,
    route_after_generation,
)
from orchestration.router_orchestrator import RouterOrchestrator
from orchestration.generation_orchestrator import GenerationOrchestrator


def build_graph(generation_llm, router_llm, rewrite_llm):
    """
    Initializes orchestrators and nodes, then constructs the state graph.
    Defines the entry point and conditional routing logic between graph nodes.
    Returns the compiled graph ready for execution.
    """

    router = RouterOrchestrator(router_llm)
    generation = GenerationOrchestrator(generation_llm)

    rewrite_node = make_rewrite_node(rewrite_llm)
    router_node = make_router_node(router)
    generation_node = make_generation_node(generation)

    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("router", router_node)
    builder.add_node("router_guardrail", router_guardrail_node)
    builder.add_node("post_router", post_router_node)

    builder.add_node("retrieval", retrieval_node)
    builder.add_node("retrieval_guardrail", retrieval_guardrail_node)

    builder.add_node("generation", generation_node)
    builder.add_node("clarification", clarification_node)

    # Entry
    builder.set_entry_point("rewrite")

    # Linear edges
    builder.add_edge("rewrite", "router")
    builder.add_edge("router", "router_guardrail")
    builder.add_edge("retrieval", "retrieval_guardrail")

    builder.add_conditional_edges(
        "router_guardrail",
        route_after_router_guardrail,
        {
            "clarify": "clarification",
            "post_router": "post_router",
        },
    )

    builder.add_conditional_edges(
        "post_router",
        route_after_post_router,
        {
            "clarify": "clarification",
            "retrieve": "retrieval",
        },
    )

    builder.add_conditional_edges(
        "retrieval_guardrail",
        route_after_retrieval_guardrail,
        {
            "clarify": "clarification",
            "generate": "generation",
        },
    )

    builder.add_conditional_edges(
        "generation",
        route_after_generation,
        {
            "clarify": "clarification",
            "__end__": END,
        },
    )

    builder.add_edge("clarification", END)

    return builder.compile()
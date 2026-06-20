from langgraph.graph import StateGraph, END

from graph_layer.state import GraphState
from graph_layer.nodes import (
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


def build_eval_graph(router, generation):

    router_node = make_router_node(router)
    generation_node = make_generation_node(generation)

    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("router_guardrail", router_guardrail_node)
    builder.add_node("post_router", post_router_node)

    builder.add_node("retrieval", retrieval_node)
    builder.add_node("retrieval_guardrail", retrieval_guardrail_node)

    builder.add_node("generation", generation_node)
    builder.add_node("clarification", clarification_node)

    builder.set_entry_point("router")

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
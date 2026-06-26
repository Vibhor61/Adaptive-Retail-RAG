"""
Utility functions for the retrieval layer.
Provides helper functions for computing signals and recording telemetry.
"""
from typing import List
from contracts.retrieval_contracts import RetrievalRawSignals

EMPTY_SIGNALS = RetrievalRawSignals(top_score=0.0, avg_score=0.0, score_distribution=[])

def compute_signals(score_values: List[float]) -> RetrievalRawSignals:
    """
    Computes top score, average score, and score distribution from a list of scores.
    Returns a RetrievalRawSignals object.
    """
    if not score_values:
        return EMPTY_SIGNALS
    return RetrievalRawSignals(
        top_score=max(score_values),
        avg_score=sum(score_values) / len(score_values),
        score_distribution=score_values,
    )

def record_retrieval_telemetry(span, score_values: List[float], result_count: int) -> None:
    """
    Records common retrieval telemetry attributes on a given span.
    """
    span.set_attribute("retrieval.result_count", result_count)
    if result_count > 0:
        span.set_attribute("retrieval.status", "success")
    else:
        span.set_attribute("retrieval.status", "miss")

    top_score = max(score_values) if score_values else 0.0

    span.set_attribute("retrieval.top_score", top_score)

    span.set_attribute(
        "retrieval.strength",
        "strong" if top_score > 0.7 else "medium" if top_score > 0.3 else "weak"
    )

    span.set_attribute("retrieval.signal_density", len(score_values))

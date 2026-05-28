from opentelemetry import trace

from .retrieval_models import (
    RetrievalBundle,
    RetrievalExecutionStatus,
    RetrievalQualityStatus,
    RetrievalEvaluationSignals,
    RetrievalEvaluationBundle
)

POSTGRES_MIN_TOP_SCORE = 0.08
POSTGRES_MIN_AVG_SCORE = 0.03

FTS_MIN_TOP_SCORE = 0.08
FTS_MIN_AVG_SCORE = 0.03

DENSE_MIN_TOP_SCORE = 0.65
DENSE_MIN_AVG_SCORE = 0.50

FUSION_MIN_TOP_SCORE = 0.015
FUSION_MIN_AVG_SCORE = 0.008

MIN_TOP_GAP = 0.02

tracer = trace.get_tracer(__name__)

def compute_score_variance(scores: list[float]) -> float:

    if len(scores) < 2:
        return 0.0
    
    avg = sum(scores)/len(scores)

    return sum([(s - avg) ** 2 for s in scores]) / (len(scores) - 1)


def compute_top_gap(scores: list[float]) -> float:

    if len(scores) < 2:
        return 0.0
    
    sorted_scores = sorted(scores, reverse=True)

    return sorted_scores[0] - sorted_scores[1]


def get_thresholds(retrieval_type: str) -> tuple[float, float]:

    if retrieval_type == "sparse_product":
        return (POSTGRES_MIN_TOP_SCORE, POSTGRES_MIN_AVG_SCORE)

    if retrieval_type == "review_fts":
        return (FTS_MIN_TOP_SCORE, FTS_MIN_AVG_SCORE)
    
    if retrieval_type == "dense_review":
        return (DENSE_MIN_TOP_SCORE, DENSE_MIN_AVG_SCORE)
    
    if retrieval_type == "fusion_review":
        return (FUSION_MIN_TOP_SCORE, FUSION_MIN_AVG_SCORE)
    
    return (0.0, 0.0)


def evaluate_retrieval(bundle: RetrievalBundle) -> RetrievalEvaluationBundle :

    with tracer.start_as_current_span("evaluate_retrieval") as span:

        anomaly_flags = []

        # Execution Failure 
        if bundle.execution_status == RetrievalExecutionStatus.FAILURE:
            signals = RetrievalEvaluationSignals(
                retrieval_type=bundle.retrieval_type,
                total_items=0,
                top_score=0.0,
                avg_score=0.0,
                score_variance=0.0,
                unique_asins=0,
            )

            return RetrievalEvaluationBundle(
                bundle=bundle,
                quality_status=RetrievalQualityStatus.FAILED,
                confidence=0.0,
                signals=signals,
                anomaly_flags=["retrieval_execution_failure"],
            )
        

        # Empty Retrieval 
        if not bundle.items:
            signals = RetrievalEvaluationSignals(
                retrieval_type=bundle.retrieval_type,
                total_items=0,
                top_score=0.0,
                avg_score=0.0,
                score_variance=0.0,
                unique_asins=0,
            )

            return RetrievalEvaluationBundle(
                bundle=bundle,
                quality_status=RetrievalQualityStatus.EMPTY,
                confidence=0.0,
                signals=signals,
                anomaly_flags=["empty_retrieval"],
            )
        

        scores = [item.score for item in bundle.items]

        avg_score = sum(scores) / len(scores)

        top_score = max(scores)

        score_variance = compute_score_variance(scores)

        top_gap = compute_top_gap(scores)
        
        unique_asins = len(
            set( item.asin for item in bundle.items if item.asin )
        )

        signals = RetrievalEvaluationSignals(
            retrieval_type=bundle.retrieval_type,
            total_items=len(bundle.items),
            top_score=top_score,
            avg_score=avg_score,
            score_variance=score_variance,
            unique_asins=unique_asins,
        )

        min_top_score, min_avg_score = (get_thresholds(bundle.retrieval_type))

        if top_score < min_top_score:
            anomaly_flags.append("low_top_score")

        if avg_score < min_avg_score:
            anomaly_flags.append("low_average_score")

        if top_gap < MIN_TOP_GAP:
            anomaly_flags.append("low_rank_separation")

        if anomaly_flags:
            status = RetrievalQualityStatus.WEAK
        else:
            status = RetrievalQualityStatus.HEALTHY


        confidence = min(
            (0.50*top_score + 0.30*avg_score + 0.20*top_gap), 1
        )


        span.set_attribute(
            "retrieval.type", bundle.retrieval_type
        )

        span.set_attribute(
            "retrieval.total_items", len(bundle.items)
        )

        span.set_attribute(
            "retrieval.top_score", top_score
        )

        span.set_attribute(
            "retrieval.avg_score", avg_score
        )

        span.set_attribute(
            "retrieval.top_gap", top_gap
        )

        span.set_attribute(
            "retrieval.status", status.value
        )

        

        return RetrievalEvaluationBundle(
            bundle=bundle,
            quality_status=status,
            confidence=confidence,
            signals=signals,
            anomaly_flags=anomaly_flags,
        )
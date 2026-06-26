"""
Provides validation and evaluation logic for retrieval bundles.
Analyzes retrieval results to determine quality status, execution success, and potential anomalies.
"""
import logging
from opentelemetry import trace

from contracts.retrieval_contracts import (
    RetrievalBundle,
    RetrievalExecutionStatus,
    RetrievalQualityStatus,
    RetrievalEvaluationSignals,
    RetrievalEvaluationBundle,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

def evaluate_retrieval(bundle: RetrievalBundle) -> RetrievalEvaluationBundle:
    """
    Evaluates a raw retrieval bundle to assess its quality and flag any anomalies.
    Returns a RetrievalEvaluationBundle containing signals and execution status.
    """
    with tracer.start_as_current_span("evaluate_retrieval") as span:
        anomaly_flags: list[str] = []

        if bundle.execution_status == RetrievalExecutionStatus.FAILURE:
            signals = RetrievalEvaluationSignals(
                retrieval_type=bundle.retrieval_type,
                total_items=0,
                unique_asins=0,
            )

            return RetrievalEvaluationBundle(
                bundle=bundle,
                query=bundle.query,
                quality_status=RetrievalQualityStatus.FAILED,
                signals=signals,
                anomaly_flags=["retrieval_execution_failure"],
            )

        total_items = len(bundle.items)

        if total_items == 0:
            signals = RetrievalEvaluationSignals(
                retrieval_type=bundle.retrieval_type,
                total_items=0,
                unique_asins=0,
            )

            return RetrievalEvaluationBundle(
                bundle=bundle,
                query=bundle.query,
                quality_status=RetrievalQualityStatus.EMPTY,
                signals=signals,
                anomaly_flags=["empty_retrieval"],
            )

        unique_asins = len(
            {item.asin for item in bundle.items if item.asin is not None}
        )

        signals = RetrievalEvaluationSignals(
            retrieval_type=bundle.retrieval_type,
            total_items=total_items,
            unique_asins=unique_asins,
        )

        status = RetrievalQualityStatus.HEALTHY
        rtype = bundle.retrieval_type

        if rtype == "sparse_product":
            if total_items == 1:
                anomaly_flags.append("single_product_match")

        elif rtype in ("review_fts", "dense_review", "fusion_review"):
            if total_items == 1:
                anomaly_flags.append("limited_review_evidence")

            if unique_asins == 1 and total_items >= 4:
                anomaly_flags.append("single_asin_bias")

        elif rtype == "candidate_gen":
            if unique_asins == 1:
                anomaly_flags.append("single_candidate")

            elif unique_asins < 3:
                anomaly_flags.append("low_candidate_diversity")

        span.set_attribute("retrieval.type", bundle.retrieval_type)
        span.set_attribute("retrieval.total_items", total_items)
        span.set_attribute("retrieval.unique_asins", unique_asins)
        span.set_attribute("retrieval.status", status.value)
        span.set_attribute("retrieval.anomaly_count", len(anomaly_flags))
        span.set_attribute("retrieval.anomalies", str(anomaly_flags))

        return RetrievalEvaluationBundle(
            bundle=bundle,
            query=bundle.query,
            quality_status=status,
            signals=signals,
            anomaly_flags=anomaly_flags,
        )
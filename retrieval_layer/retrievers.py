import logging
import psycopg2
import json

from typing import Optional
from qdrant_client import QdrantClient
from opentelemetry import trace
from sentence_transformers import SentenceTransformer

from config.settings import settings
from contracts.retrieval_contracts import (
    RetrievalResult,
    RetrievalBundle,
    RetrievalExecutionStatus,
    RetrievalRawSignals
)

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": settings.postgres_host,
    "database": settings.postgres_db,
    "user": settings.postgres_user,
    "password": settings.postgres_password,
    "port": settings.postgres_port
}

QDRANT_HOST = settings.qdrant_host
QDRANT_PORT = settings.qdrant_port
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

tracer = trace.get_tracer(__name__)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBEDDING_MODEL)

EMPTY_SIGNALS = RetrievalRawSignals(top_score=0.0, avg_score=0.0, score_distribution=[])


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def _safe_json(val):
    try:
        if val is None:
            return ""
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)
    except Exception:
        return ""
    

def compute_signals(score_values: list[float]) -> RetrievalRawSignals:
    if not score_values:
        return EMPTY_SIGNALS
    return RetrievalRawSignals(
        top_score=max(score_values),
        avg_score=sum(score_values) / len(score_values),
        score_distribution=score_values,
    )

def sparse_fact_retrieval(entity: Optional[str]=None, query: Optional[str]=None, top_k: int = 5,) -> RetrievalBundle:
    
    identifier = entity or query
    if not identifier:
        raise ValueError("sparse_fact_retrieval requires either entity or query")

    with tracer.start_as_current_span("sparse_fact_retrieval") as span:
        span.set_attribute("retrieval.identifier", identifier)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "postgres_fts")

        try: 
            sql_query = """
                SELECT asin, title, brand, category, main_cat, description, feature, price, price_raw,
                    ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
                FROM products_table
                WHERE search_vector @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s;
            """

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query, (identifier, identifier, top_k))
                    results = cur.fetchall()

            retrieval_results = []
            for rank, row in enumerate(results, start=1):
                asin, title, brand, category, main_cat, description, feature, price, price_raw, score = row

                text = f"""
                    PRODUCT:
                    Title: {title}
                    Brand: {brand}
                    Category: {main_cat}
                    Price: {price}

                    Description:
                    {description}
                    Features:
                    {feature}
                    """.strip()

                retrieval_results.append(
                    RetrievalResult(
                        source="sparse_product",
                        doc_id=asin,
                        asin=asin,
                        text=text,  
                        score=float(score),
                        rank=rank,
                        metadata={
                            "title": title,
                            "brand": brand,
                            "category": category,
                            "main_cat": main_cat,
                            "description": description,
                            "feature": feature,
                            "price": price,
                            "price_raw": price_raw,
                        },
                    )
                )

            score_values = [float(row[9]) for row in results] 
            span.set_attribute("retrieval.result_count", len(retrieval_results))
            
            if retrieval_results:
                span.set_attribute("retrieval.status", "success")
            else:
                span.set_attribute("retrieval.status",  "miss")

            top_score = max(score_values) if score_values else 0.0

            span.set_attribute("retrieval.top_score", top_score)

            span.set_attribute(
                "retrieval.strength",
                "strong" if top_score > 0.7 else "medium" if top_score > 0.3 else "weak"
            )

            span.set_attribute("retrieval.signal_density", len(score_values))

            return RetrievalBundle(
                entity=entity,
                query=query,
                retrieval_type="sparse_product",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=retrieval_results,
                raw_signals=compute_signals(score_values),
            )

        except Exception as e:
            span.set_attribute("retrieval.status", "error")
            span.set_attribute("retrieval.result_count", 0)
            span.record_exception(e)
            logger.exception("sparse_fact_retrieval failed")
            raise


def review_fts_retrieval(query: str, top_k: int = 5,) -> RetrievalBundle:

    with tracer.start_as_current_span("review_fts_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "review_fts")

        try:
            sql_query = """
                SELECT asin, review_id, summary_text, review_text,
                    ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
                FROM reviews_table
                WHERE search_vector @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s;
            """

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query, (query, query, top_k))
                    results = cur.fetchall()

            retrieval_results = []
            for rank, row in enumerate(results, start=1):
                summary     = row[2] or ""
                review_text = row[3] or ""
                combined    = f"""
                    REVIEW:
                    Summary:
                    {summary}
                    Text:
                    {review_text}""".strip()

                retrieval_results.append(
                    RetrievalResult(
                        source="review_fts",
                        doc_id=row[0],
                        review_id=row[1],
                        asin=row[0],
                        score=float(row[4]),
                        rank=rank,
                        text=combined,
                        metadata={},
                    )
                )

            score_values = [float(row[4]) for row in results]
            span.set_attribute("retrieval.result_count", len(retrieval_results))
            
            if retrieval_results:
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

            return RetrievalBundle(
                query=query,
                retrieval_type="review_fts",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=retrieval_results,
                raw_signals=compute_signals(score_values),
            )

        except Exception as e:
            span.set_attribute("retrieval.status", "error")
            span.set_attribute("retrieval.result_count", 0)
            span.record_exception(e)
            logger.exception("review_fts_retrieval failed")
            raise


def dense_review_retrieval(query: str, top_k: int = 5,) -> RetrievalBundle:

    with tracer.start_as_current_span("dense_review_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "qdrant_dense")

        try:
            query_embedding = embedder.encode(query).tolist()

            search_result = qdrant_client.query_points(
                collection_name="reviews_embeddings",
                query=query_embedding,
                limit=top_k,
            )

            points = search_result.points
            retrieval_results = []

            for rank, item in enumerate(points, start=1):
                payload = item.payload or {}
                retrieval_results.append(
                RetrievalResult(
                    source="dense_review",
                    doc_id=str(item.id),
                    review_id=payload.get("review_id"),
                    asin=payload.get("asin"),
                    text=payload.get("text", ""),
                    score=float(item.score),
                    rank=rank,
                    metadata=payload,
                )
            )

            score_values = [float(item.score) for item in points]
            span.set_attribute("retrieval.result_count", len(retrieval_results))

            if retrieval_results:
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

            return RetrievalBundle(
                query=query,
                retrieval_type="dense_review",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=retrieval_results,
                raw_signals=compute_signals(score_values),
            )

        except Exception as e:
            span.set_attribute("retrieval.status", "error")
            span.set_attribute("retrieval.result_count", 0)
            span.record_exception(e)
            logger.exception("dense_review_retrieval failed")
            raise


def fusion_retrieval(query: str, top_k: int = 5, fusion_k: int = 60,) -> RetrievalBundle:

    with tracer.start_as_current_span("fusion_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("fusion.type", "review_fts + dense")
        span.set_attribute("fusion.top_k", top_k)
        span.set_attribute("fusion.k", fusion_k)

        try:
            fts_bundle   = review_fts_retrieval(query=query, top_k=top_k)
            dense_bundle = dense_review_retrieval(query=query, top_k=top_k)

            scores    = {}
            best_item = {}

            for item in (fts_bundle.items + dense_bundle.items):
                if item.rank is None:
                    continue
                rrf_score = 1.0 / (fusion_k + item.rank)
                key = f"{item.source}:{item.doc_id}"
                scores[key] = scores.get(key, 0.0) + rrf_score
                if key not in best_item or item.rank < best_item[key].rank:
                    best_item[key] = item

            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            fused_results = []
            score_values  = []

            for final_rank, (key, score) in enumerate(ordered, start=1):
                base = best_item[key].model_copy(update={
                    "rank": final_rank,
                    "score": float(score),
                    "metadata": {
                        **dict(best_item[key].metadata or {}),
                        "fusion_score": float(score),
                        "fusion_k": fusion_k,
                        "fusion_sources": "review_fts+dense",
                    }
                })
                score_values.append(float(score))
                fused_results.append(base)

            span.set_attribute("retrieval.result_count", len(fused_results))

            if fused_results:
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

            return RetrievalBundle(
                query=query,
                retrieval_type="fusion_review",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=fused_results,
                raw_signals=compute_signals(score_values),
            )

        except Exception as e:
            span.set_attribute("retrieval.status", "error")
            span.set_attribute("retrieval.result_count", 0)
            span.record_exception(e)
            logger.exception("fusion_retrieval failed")
            raise


def candidate_gen_retrieval(query: str,top_k: int = 5) -> RetrievalBundle:

    with tracer.start_as_current_span("candidate_gen_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "candidate_gen")

        try:
            sparse_bundle = sparse_fact_retrieval(query=query, top_k=20)
            fts_bundle = review_fts_retrieval(query=query, top_k=20)

            seen: set[str] = set()
            deduplicated: list[RetrievalResult] = []

            sparse_items = sparse_bundle.items[:top_k]
            candidate_asins = {item.asin for item in sparse_items if item.asin}

            # filter reviews to only matching ASINs
            matched_reviews = [item for item in fts_bundle.items if item.asin in candidate_asins]

            candidates = sparse_items + matched_reviews
            score_values = [item.score for item in candidates]

            span.set_attribute("retrieval.result_count", len(candidates))

            if candidates:
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
            
            return RetrievalBundle(
                query=query,
                retrieval_type="candidate_gen",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=candidates,
                raw_signals=compute_signals(score_values),
            )

        except Exception as e:
            span.set_attribute("retrieval.status", "error")
            span.set_attribute("retrieval.result_count", 0)
            span.record_exception(e)
            logger.exception("candidate_gen_retrieval failed")
            raise
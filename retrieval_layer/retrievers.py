import logging
import psycopg2
import re

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

EMBEDDING_MODEL = settings.embedding_model

tracer = trace.get_tracer(__name__)
qdrant_client = QdrantClient(url=settings.qdrant_url)
embedder = SentenceTransformer(EMBEDDING_MODEL)

EMPTY_SIGNALS = RetrievalRawSignals(top_score=0.0, avg_score=0.0, score_distribution=[])

def get_connection():
    return psycopg2.connect(settings.postgres_url)


def compute_signals(score_values: list[float]) -> RetrievalRawSignals:
    if not score_values:
        return EMPTY_SIGNALS
    return RetrievalRawSignals(
        top_score=max(score_values),
        avg_score=sum(score_values) / len(score_values),
        score_distribution=score_values,
    )

def sparse_fact_retrieval(entity: Optional[str]=None, query: Optional[str]=None, top_k: int = 5, asin: Optional[str] = None) -> RetrievalBundle:
    
    identifier = entity or query
    if not identifier and not asin:
        raise ValueError("sparse_fact_retrieval requires asin, entity, or query")
 
    with tracer.start_as_current_span("sparse_fact_retrieval") as span:
        span.set_attribute("retrieval.identifier", identifier or "")
        span.set_attribute("retrieval.asin", asin or "")
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "postgres_asin_lookup" if asin else "postgres_fts")
 
        try:
            if asin:
                sql_query = """
                    SELECT asin, title, brand, category, main_cat, description, feature, price, price_raw,
                        1.0 AS score
                    FROM products_table
                    WHERE asin = %s
                    LIMIT %s;
                """
                params = (asin, top_k)
            else:
                sql_query = """
                    SELECT asin, title, brand, category, main_cat, description, feature, price, price_raw,
                        ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
                    FROM products_table
                    WHERE search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s;
                """
                params = (identifier, identifier, top_k)
 
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query, params)
                    results = cur.fetchall()
 
            retrieval_results = []
            for rank, row in enumerate(results, start=1):
                row_asin, title, brand, category, main_cat, description, feature, price, price_raw, score = row
 
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
                        doc_id=row_asin,
                        asin=row_asin,
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
                span.set_attribute("retrieval.status", "miss")
 
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
                SELECT r.asin, r.review_id, r.summary_text, r.review_text,
                    p.title, p.brand, p.price,
                    ts_rank_cd(r.search_vector, websearch_to_tsquery('english', %s)) AS score
                FROM reviews_table r
                LEFT JOIN products_table p ON r.asin = p.asin
                WHERE r.search_vector @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s;
            """

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query, (query, query, top_k))
                    results = cur.fetchall()

            retrieval_results = []
            for rank, row in enumerate(results, start=1):
                asin = row[0]
                review_id = row[1]
                summary = row[2] or ""
                review_text = row[3] or ""
                title = row[4] or "Unknown Product"
                brand = row[5] or ""
                price = row[6] or ""

                combined = f"""
                    PRODUCT: {title} | Brand: {brand} | Price: {price}
                    REVIEW:
                    Summary: {summary}
                    Text: {review_text}
                """.strip()

                retrieval_results.append(
                    RetrievalResult(
                        source="review_fts",
                        doc_id=row[0],
                        review_id=row[1],
                        asin=row[0],
                        score=float(row[7]),
                        rank=rank,
                        text=combined,
                        metadata={"title": title, "brand": brand},
                    )
                )

            score_values = [float(row[7]) for row in results]
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

            asins = [p.payload.get("asin") for p in points if p.payload and p.payload.get("asin")]

            title_map = {}
            if asins:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT asin, title, brand, price FROM products_table WHERE asin = ANY(%s)",
                            (asins,)
                        )
                        title_map = {row[0]: (row[1], row[2], row[3]) for row in cur.fetchall()}

            retrieval_results = []

            for rank, item in enumerate(points, start=1):
                payload = item.payload or {}
                asin = payload.get("asin")
                title, brand, price = title_map.get(asin, ("Unknown Product", "", ""))

                combined = f"""
                    PRODUCT: {title} | Brand: {brand} | Price: {price}
                    REVIEW:
                    Text: {payload.get('text', '')}
                """.strip()
                retrieval_results.append(
                RetrievalResult(
                    source="dense_review",
                    doc_id=str(item.id),
                    review_id=payload.get("review_id"),
                    asin=asin,
                    text=combined,
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
            fts_bundle = review_fts_retrieval(query=query, top_k=top_k)
            dense_bundle = dense_review_retrieval(query=query, top_k=top_k)

            scores = {}
            best_item = {}

            span.set_attribute(
                "fusion.fts_count", len(fts_bundle.items)
            )

            span.set_attribute(
                "fusion.dense_count", len(dense_bundle.items)
            )
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
            score_values = []

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
            # ----------------------------
            # STEP 1: HARD QUERY ENTITY PRESERVATION (CRITICAL FIX)
            # ----------------------------
            query_lower = query.lower()

            # Extract device-like tokens (simple but stable heuristic)
            device_tokens = set(re.findall(r'\b[a-z]*\d+[a-z]*\b|\biphone\b|\bsamsung\b|\bhuawei\b', query_lower))

            # Remove generic noise but DO NOT delete structure aggressively
            SOFT_NOISE = {
                "recommend", "suggest", "show", "find", "looking",
                "for", "with", "a", "an", "or", "and"
            }

            cleaned_query_tokens = [
                t for t in re.findall(r'\b\w+\b', query_lower)
                if t not in SOFT_NOISE
            ]

            entity_query = " ".join(cleaned_query_tokens)

            # If we have device signal, preserve it explicitly
            if device_tokens:
                entity_query = " ".join(sorted(device_tokens)) + " " + entity_query

            span.set_attribute("retrieval.entity_query", entity_query)

            # ----------------------------
            # STEP 2: PARALLEL RETRIEVAL (UNCHANGED BUT STABLE INPUT)
            # ----------------------------
            sparse_bundle = sparse_fact_retrieval(query=entity_query, top_k=50)
            fts_bundle = review_fts_retrieval(query=query, top_k=50)

            # ----------------------------
            # STEP 3: CANDIDATE BUILD (MAJOR FIX)
            # ----------------------------

            # 3.1 Primary candidate pool = sparse retrieval ONLY
            primary_items = sparse_bundle.items[:top_k]

            # 3.2 Build ASIN whitelist from primary only
            candidate_asins = {item.asin for item in primary_items if item.asin}

            # 3.3 Attach review signals ONLY for existing candidates
            review_items = [
                item for item in fts_bundle.items
                if item.asin in candidate_asins
            ]

            # ----------------------------
            # STEP 4: SCORE NORMALIZATION (FIXED BIAS)
            # ----------------------------
            def normalize(item):
                # sparse is primary signal
                if item in primary_items:
                    item.score = item.score * 1.0
                else:
                    # review boost is capped
                    item.score = item.score * 0.3
                return item

            primary_items = [normalize(i) for i in primary_items]
            review_items = [normalize(i) for i in review_items]

            candidates = primary_items + review_items

            # ----------------------------
            # STEP 5: NO SELF-DESTRUCTIVE QUERY RELAXATION (IMPORTANT FIX)
            # ----------------------------
            if not candidates:
                fallback_query = " ".join(device_tokens) if device_tokens else query_lower
                span.set_attribute("retrieval.fallback_query", fallback_query)

                sparse_bundle = sparse_fact_retrieval(query=fallback_query, top_k=top_k)
                candidates = sparse_bundle.items

            # ----------------------------
            # STEP 6: METRICS
            # ----------------------------
            scores = [c.score for c in candidates]

            span.set_attribute("retrieval.result_count", len(candidates))
            span.set_attribute("retrieval.status", "success" if candidates else "miss")

            span.set_attribute(
                "retrieval.strength",
                "strong" if max(scores, default=0) > 0.7 else
                "medium" if max(scores, default=0) > 0.3 else "weak"
            )

            return RetrievalBundle(
                query=query,
                retrieval_type="candidate_gen_fixed",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=candidates,
                raw_signals=compute_signals(scores),
            )

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("retrieval.status", "error")
            raise
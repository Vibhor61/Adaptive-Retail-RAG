import os
import psycopg2
from copy import deepcopy

from qdrant_client import QdrantClient
from opentelemetry import trace
from sentence_transformers import SentenceTransformer
from ..contracts.retrieval_contracts import (
    RetrievalResult,
    RetrievalBundle,
    RetrievalExecutionStatus,
    RetrievalRawSignals
)

DB_CONFIG = {
    "host" : os.getenv("POSTGRES_HOST"),
    "database" : os.getenv("POSTGRES_DB"),
    "user" : os.getenv("POSTGRES_USER"),
    "password" : os.getenv("POSTGRES_PASSWORD"),
    "port" : int(os.getenv("POSTGRES_PORT"))
}

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

tracer = trace.get_tracer(__name__)


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# Replace raw query here with grounded entities after stabilization
def sparse_fact_retrieval(query: str, top_k: int = 5) -> RetrievalBundle:
    
    with tracer.start_as_current_span("sparse_fact_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "postgres_bm25")
        

        try:
            conn = get_connection()
            with conn.cursor() as cur:
                sql_query = """
                    SELECT asin, title, brand, category, price, price_raw,
                        ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
                    FROM products_table
                    WHERE search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s;
                """
                
                cur.execute(sql_query, (query, query, top_k))
                results = cur.fetchall() 

            retrieval_results = []

            

            for rank, row in enumerate(results):
                retrieval_results.append(RetrievalResult(
                    source="sparse",
                    doc_id=row[0],
                    review_id=None,
                    asin=row[0],
                    text=f"{row[1]} {row[2]} {row[3]} {row[4]} {row[5]}",
                    score=row[6],
                    rank=rank,
                    metadata={"title": row[1], "brand": row[2], "category": row[3], "price": row[4], "price_raw": row[5]}
                ))
            
            span.set_attribute("retrieval.hit_count", len(retrieval_results))
            
            score_values = [float(row[6]) for row in results] if results else []
            top_score = max(score_values) if score_values else 0.0
            avg_score = sum(score_values)/len(score_values) if score_values else 0.0

            raw_signals = RetrievalRawSignals(
                top_score=top_score,
                avg_score=avg_score,
                score_distribution=score_values
            )

            return RetrievalBundle(
                query=query,
                retrieval_type="sparse_product",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=retrieval_results,
                raw_signals=raw_signals
            )
        
        except Exception as e:

            span.record_exception(e)

            return RetrievalBundle(
                query=query,
                retrieval_type="sparse_product",
                execution_status=RetrievalExecutionStatus.FAILURE,
                items=[],
                raw_signals=raw_signals,
                failure_reason=str(e)
            )


def review_fts_retrieval(query: str, top_k: int = 5) -> RetrievalBundle:

    with tracer.start_as_current_span("review_fts_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "review_fts")

        try:

            conn = get_connection() 
            with conn.cursor() as cur:
                sql_query = """
                    SELECT asin, review_id, summary_text, review_text,
                        ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
                    FROM reviews_table
                    WHERE search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                """

                cur.execute(sql_query, (query, query, top_k))
                results = cur.fetchall()

            retrieval_results = []

            for rank, row in enumerate(results):

                summary = row[2] or ""
                review_text = row[3] or ""
                combined_text = (
                    f"{summary}\n{review_text}"
                ).strip()

                retrieval_results.append(
                    RetrievalResult(
                        source="review_fts",
                        doc_id=row[0],
                        review_id=row[1],
                        asin=row[0],
                        score=float(row[4]),
                        rank=rank,
                        text=combined_text,
                        metadata={},
                    )
                )

            span.set_attribute("retrieval.hit_count",len(retrieval_results),)

            score_values = [float(row[4]) for row in results] if results else [] 
            top_score = sum(score_values) if score_values else 0.0
            avg_score = sum(score_values)/len(score_values) if score_values else 0.0

            raw_signals = RetrievalRawSignals(
                top_score=top_score,
                avg_score=avg_score,
                score_distribution=score_values
            )

            return RetrievalBundle(
                query=query,
                retrieval_type="review_fts",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=retrieval_results,
                raw_signals=raw_signals
            )

        except Exception as e:

            span.record_exception(e)

            return RetrievalBundle(
                query=query,
                retrieval_type="review_fts",
                execution_status=RetrievalExecutionStatus.FAILURE,
                items=[],
                raw_signals=raw_signals,
                failure_reason=str(e)
            )


def dense_review_retrieval(query: str, top_k: int = 5) -> RetrievalBundle:
    
    with tracer.start_as_current_span("dense_review_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "qdrant_dense")
        
        try:

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            model = SentenceTransformer(EMBEDDING_MODEL)
            query_embedding = model.encode(query).tolist()
            
            search_result = client.search(
                collection_name="reviews_embeddings",
                query_vector=query_embedding,
                limit=top_k
            )
            
            retrieval_results = []
            for rank, item in enumerate(search_result):

                payload = item.payload or {}

                retrieval_results.append(RetrievalResult(
                    source="dense_review",
                    doc_id=str(item.id),
                    review_id=payload.get("review_id"),
                    asin=payload.get("asin"),
                    text=payload.get("text",""),
                    score=float(item.score),
                    rank=rank,
                    metadata=payload
                ))

            span.set_attribute("retrieval.hit_count", len(retrieval_results))

            score_values = [float(row[4]) for row in search_result] if search_result else [] 
            top_score = sum(score_values) if score_values else 0.0
            avg_score = sum(score_values)/len(score_values) if score_values else 0.0

            raw_signals = RetrievalRawSignals(
                top_score=top_score,
                avg_score=avg_score,
                score_distribution=score_values
            )
            

            return RetrievalBundle(
                query=query,
                retrieval_type="dense_review",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=retrieval_results,
                raw_signals=raw_signals
            )
        
        except Exception as e:

            span.record_exception(e)

            return RetrievalBundle(
                query=query,
                retrieval_type="dense_review",
                execution_status=RetrievalExecutionStatus.FAILURE,
                items=[],
                raw_signals=raw_signals,
                failure_reason=str(e),
            )



def fusion_retrieval(query: str, top_k: int = 5, fusion_k: int = 60) -> RetrievalBundle:
    
    with tracer.start_as_current_span("fusion_review_retrieval") as span:
        
        try:
            span.set_attribute("fusion.type", "review_fts + dense")
            span.set_attribute("fusion.top_k", top_k)

            scores = {}
            best_item = {}

            fts_items = review_fts_retrieval(query, top_k)
            dense_items = dense_review_retrieval(query, top_k)

            for item in (fts_items.items + dense_items.items):
                if item.rank is None:
                    continue

                # reciprocal rank fusion score
                rrf_score = 1.0 / (fusion_k + item.rank)

                key = f"{item.source}:{item.doc_id}"

                scores[key] = scores.get(key, 0.0) + rrf_score

                if key not in best_item or item.rank < best_item[key].rank:
                    best_item[key] = item

            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
            fused_results = []
            score_values = []
            
            for final_rank, (key, score) in enumerate(ordered, start=1):

                base = deepcopy(best_item[key])

                base.rank = final_rank
                base.score = float(score)

                base.metadata = dict(base.metadata or {})
                base.metadata["fusion_score"] = float(score)
                base.metadata["fusion_k"] = fusion_k
                base.metadata["fusion_sources"] = "review_fts+dense"

                score_values.append(float(score))
                fused_results.append(base)

            span.set_attribute("fusion.output_count", len(fused_results))
            
            top_score = max(score_values) if score_values else 0.0
            avg_score = sum(score_values)/len(score_values) if score_values else 0.0

            raw_signals = RetrievalRawSignals(
                top_score=top_score,
                avg_score=avg_score,
                score_distribution=score_values
            )

            return RetrievalBundle(
                query=query,
                retrieval_type="fusion_review",
                execution_status=RetrievalExecutionStatus.SUCCESS,
                items=fused_results,
                raw_signals=raw_signals
            )
        
        except Exception as e:

            span.record_exception(e)

            return RetrievalBundle(
                query="fusion_review",
                retrieval_type="fusion_review",
                execution_status=RetrievalExecutionStatus.FAILURE,
                items=[],
                raw_signals=raw_signals,
                failure_reason=str(e)
            )
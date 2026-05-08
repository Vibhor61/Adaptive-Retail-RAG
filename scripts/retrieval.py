import os
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import psycopg2
from typing import List, Optional
from copy import deepcopy
import enum
from opentelemetry import trace
import enum


DB_CONFIG = {
    "host" : os.getenv("POSTGRES_HOST"),
    "database" : os.getenv("POSTGRES_DB"),
    "user" : os.getenv("POSTGRES_USER"),
    "password" : os.getenv("POSTGRES_PASSWORD"),
    "port" : int(os.getenv("POSTGRES_PORT"))
}

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

POSTGRES_SCORE_THRESHOLD = 0.01
QDRANT_SCORE_THRESHOLD = 0.5


tracer = trace.get_tracer(__name__)

class RetrievalStatus(enum.Enum):
    PASSED = "passed"
    EMPTY = "empty"
    LOW_QUALITY = "low_quality"
    SEMANTIC_FAILURE = "semantic_failure"


class QueryType(enum.Enum):
    PRODUCT_LOOKUP = "product_lookup"   
    REVIEW_QUERY = "review_query"     
    COMPARISON = "comparison"           
    VAGUE = "vague"   
    
@dataclass
class RetrievalSignals:
    sparse_hit_count: int
    avg_sparse_score: float        
    top_sparse_score: float

    dense_hit_count: int
    avg_dense_score: float       
    top_dense_score: float

    asin_overlap: int
    mode_used: str


@dataclass 
class RetrievalResult:
    source : str
    doc_id : str
    review_id: Optional[int]
    asin_id: str
    text: str
    score: float
    rank : int
    metadata: dict


@dataclass
class FinalResult:
    query: str
    resolved_asin: Optional[str]
    items: List[RetrievalResult]

    status: Optional[RetrievalStatus] = None
    failure_reason: Optional[str] = None
    signals: Optional[RetrievalSignals] = None


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def compute_retrieval_signals(sparse_results: Optional[FinalResult], dense_results: Optional[FinalResult], mode_used):
    sparse_scores = [r.score for r in sparse_results]
    dense_scores = [r.score for r in dense_results]
    return RetrievalSignals(
        sparse_hit_count = len(sparse_results),
        avg_sparse_score = sum(sparse_scores) / len(sparse_scores) if sparse_scores else 0.0,
        top_sparse_score = max(sparse_scores) if sparse_scores else 0.0,
        dense_hit_count = len(dense_results),
        avg_dense_score = sum(dense_scores) / len(dense_scores) if dense_scores else 0.0,
        top_dense_score = max(dense_scores) if dense_scores else 0.0,
        asin_overlap = compute_asin_overlap(sparse_results, dense_results),
        mode_used = mode_used
    )

def compute_asin_overlap(sparse_results, dense_results):
    if not sparse_results or not dense_results:
        return 0
    
    sparse_asins = set([r.asin_id for r in sparse_results])
    dense_asins = set([r.asin_id for r in dense_results])
    return len(sparse_asins & dense_asins)


def sparse_fact_retrieval(query: str, top_k: int = 5) -> FinalResult:
    
    with tracer.start_as_current_span("sparse_fact_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "postgres_bm25")
        
        conn = get_connection()
        cursor = conn.cursor()
        
        sql_query = """
            SELECT asin, title, brand, category, price, price_raw,
                ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
            FROM products_table
            WHERE search_vector @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s;"""
        
        cursor.execute(sql_query, (query, query, top_k))
        results = cursor.fetchall() 

        cursor.close()
        conn.close()

        retrieval_results = []
        for rank, row in enumerate(results):
            retrieval_results.append(RetrievalResult(
                source="sparse",
                doc_id=row[0],
                review_id=None,
                asin_id=row[0],
                text=f"{row[1]} {row[2]} {row[3]} {row[4]} {row[5]}",
                score=row[6],
                rank=rank,
                metadata={"title": row[1], "brand": row[2], "category": row[3], "price": row[4], "price_raw": row[5]}
            ))
        

        scores = [r.score for r in retrieval_results]
        span.set_attribute("retrieval.hit_count", len(retrieval_results))
        span.set_attribute("retrieval.top_score", max(scores) if scores else 0.0)
        span.set_attribute("retrieval.avg_score", sum(scores) / len(scores) if scores else 0.0)
        
        signals = compute_retrieval_signals(retrieval_results, [], "sparse")

        return FinalResult(
            query=query,
            resolved_asin=retrieval_results[0].asin_id if retrieval_results else None,
            items=retrieval_results,
            signals=signals
        )


def dense_fact_retrieval(query: str, top_k: int = 5) -> List[RetrievalResult]:
    
    with tracer.start_as_current_span("dense_fact_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.source", "qdrant_dense")
        
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        query_embedding = EMBEDDING_MODEL.encode(query).tolist()
        
        search_result = client.search(
            collection_name="reviews_embeddings",
            query_vector=query_embedding,
            limit=top_k
        )
        
        retrieval_results = []
        for rank, item in enumerate(search_result):
            retrieval_results.append(RetrievalResult(
                source="dense",
                doc_id=str(item.id),
                review_id=item.payload.get("review_id"),
                asin_id=item.payload.get("asin"),
                text=item.payload.get("text"),
                score=item.score,
                rank=rank,
                metadata={"review_id": item.payload.get("review_id"), "asin": item.payload.get("asin")}
            ))

        scores = [r.score for r in retrieval_results]
        span.set_attribute("retrieval.hit_count", len(retrieval_results))
        span.set_attribute("retrieval.top_score", max(scores) if scores else 0.0)
        span.set_attribute("retrieval.avg_score", sum(scores) / len(scores) if scores else 0.0)

        signals = compute_retrieval_signals([], retrieval_results, "dense")

        return FinalResult(
            query=query,
            resolved_asin=retrieval_results[0].asin_id if retrieval_results else None,
            items=retrieval_results,
            signals=signals
        )
    


def compute_rrf_score(item, fusion_k=60, w_sparse=0.5, w_dense=0.5):
    if item.rank is None:
        raise ValueError("Rank cannot be None for RRF score computation")
    
    if item.source == "sparse":
        weight = w_sparse
    else:
        weight = w_dense

    return weight * (1.0 / (fusion_k + item.rank))


def evaluate_retrieval(final: FinalResult) -> FinalResult:

    signals = final.signals

    if not final.items:
        final.status = RetrievalStatus.EMPTY
        final.failure_reason = "No results"
        return final

    if signals.mode_used == "sparse":
        if signals.avg_sparse_score < POSTGRES_SCORE_THRESHOLD:
            final.status = RetrievalStatus.LOW_QUALITY
            final.failure_reason = "Sparse below threshold"
        else:
            final.status = RetrievalStatus.PASSED

    elif signals.mode_used == "dense":
        if signals.avg_dense_score < QDRANT_SCORE_THRESHOLD:
            final.status = RetrievalStatus.SEMANTIC_FAILURE
            final.failure_reason = "Low semantic similarity"
        else:
            final.status = RetrievalStatus.PASSED

    else:  # hybrid
        sparse_strong = signals.avg_sparse_score >= POSTGRES_SCORE_THRESHOLD
        dense_strong = signals.avg_dense_score >= QDRANT_SCORE_THRESHOLD

        if signals.asin_overlap > 0:
            final.status = RetrievalStatus.PASSED

        elif sparse_strong or dense_strong:
            final.status = RetrievalStatus.PASSED

        else:
            final.status = RetrievalStatus.LOW_QUALITY
            final.failure_reason = "Weak signals and no agreement"

    return final


def fusion_retrieval(query: str, top_k: int = 5, k :int =60, strategy:str="hybrid") -> FinalResult:
    
    with tracer.start_as_current_span("fusion_retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.rrf_k", k)
        span.set_attribute("retrieval.strategy", strategy)

        sparse_results = sparse_fact_retrieval(query, top_k) 

        dense_results = dense_fact_retrieval(query, top_k)

        scores = {}
        best_asin = {}

        for item in sparse_results.items + dense_results.items:
            if item.rank is None:
                raise ValueError("Rank cannot be None")
            key = f"{item.source}:{item.doc_id}"
            rrf_score = compute_rrf_score(item, fusion_k=k, w_sparse=0.5, w_dense=0.5)
            scores[key] = scores.get(key, 0) + rrf_score
            if key not in best_asin or best_asin[key].rank > item.rank:
                best_asin[key] = item

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        fused_results = []
        for final_rank, (key, score) in enumerate(ordered, start=1):
            base = best_asin[key]
            copied = deepcopy(base)
            # Only set new rank and score, do not overwrite original info in metadata
            copied.rank = final_rank
            copied.score = score
            copied.metadata = dict(copied.metadata or {})
            copied.metadata["rrf_score"] = float(score)
            copied.metadata["rrf_k"] = k
            fused_results.append(copied)

        # Unify stats and signals
        signals = compute_retrieval_signals(sparse_results.items, dense_results.items, strategy)

        span.set_attribute("fusion.sparse_hits", signals.sparse_hit_count)
        span.set_attribute("fusion.dense_hits", signals.dense_hit_count)
        span.set_attribute("fusion.asin_overlap", signals.asin_overlap)
        span.set_attribute("fusion.fused_count", len(fused_results))
        
        return FinalResult(
            query=query,
            resolved_asin=fused_results[0].asin_id if fused_results else None,
            items=fused_results,
            signals=signals
        )
    

def retreive(query: str, mode: str) -> FinalResult:

    if mode == "sparse":
        final = sparse_fact_retrieval(query)

    elif mode == "dense":
        final = dense_fact_retrieval(query)

    else:
        final = fusion_retrieval(query)

    return evaluate_retrieval(final)


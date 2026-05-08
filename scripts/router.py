import json
import re
import os
import psycopg2
from dataclasses import dataclass
from langchain_ollama import OllamaLLM
import enum
from typing import Optional, Set
import re
from opentelemetry import trace
from phoenix_connection import trace_db_query
tracer = trace.get_tracer(__name__)

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST","postgres"),
    "database": os.getenv("POSTGRES_DB","rag_db"),
    "user": os.getenv("POSTGRES_USER","rag_user"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT",5432))
}

FUZZY_THRESHOLD = 90
MAX_PHRASE_LENGTH = 4
MIN_PHRASE_CHARS = 3 

SPARSE_KEYWORDS = {"price", "cost", "brand", "category", "cheap", "expensive", "how much", "rate"}
DENSE_KEYWORDS = {"best", "comfortable", "recommend", "review", "good", "worst", "experience", "quality", "feel", "worth"}
HYBRID_KEYWORDS = {"compare", "vs", "versus", "difference", "better", "between"}

KNOWN_CATEGORIES = set()
KNOWN_BRANDS = set()

router_llm = OllamaLLM(model="mistral:latest")

class RouterStatus(enum.Enum):
    Passed = "passed"
    Low_Confidence = "low_confidence"
    Vague = "vague"
    Ambiguous = "ambiguous"


class QueryType(enum.Enum):
    PRODUCT_LOOKUP = "product_lookup"   
    REVIEW_QUERY = "review_query"     
    COMPARISON = "comparison"           
    VAGUE = "vague"                     


@dataclass
class RouterResult:
    retrieval_type: str
    query_type: QueryType
    status: RouterStatus

    confidence: float
    reason: str
    signals: dict

    llm_used: bool 
    entity_signal: Optional[str] = None
    failure_reason: Optional[str] = None


class DBEntityLoader:
    def connect(self):
        return psycopg2.connect(**DB_CONFIG)
    
    @trace_db_query
    def exact_match(self, field:str, value:str) -> tuple[str|None, float]:
        query = f"SELECT 1 FROM products_table WHERE {field} = %s LIMIT 1"
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query, (value,))
            row = cur.fetchone() 
        
        if row:
            return row[0], 1.0

        return None, 0.0

    @trace_db_query
    def fuzzy_match(self, field:str, value:str) -> tuple[str|None, float]:
        query = f"""
            SELECT {field} , similarity(%s, {field}) as score 
            FROM products_table WHERE {field} IS NOT NULL 
            ORDER BY score DESC 
            LIMIT 1"
        """
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query, (value,))
            row = cur.fetchone()
        
        if row:
            score = float(row[1])
            return row[0], score
        
        return None, 0.0

    def resolve(self, field: str, value: str, threshold: float = 0.75):
        exact, score = self.exact_match(field, value)
        if exact:
            return exact, score, "exact"

        fuzzy, score = self.fuzzy_match(field, value)
        if fuzzy and score >= threshold:
            return fuzzy, score, "fuzzy"

        return None, 0.0, "none"

resolver = DBEntityLoader()


def preprocess_query(query:str) -> list[str]:
    query = query.lower()
    query = re.sub(r'[^\w\s]', ' ', query)
    return query.split()


# It will be used for isolating query type for interpretability after Evaluation
def map_query_type(retrieval_type:str, tokens: Set[str]) -> QueryType:
    if tokens & HYBRID_KEYWORDS:
        return QueryType.COMPARISON
    if tokens & DENSE_KEYWORDS:
        return QueryType.REVIEW_QUERY
    if tokens & SPARSE_KEYWORDS:
        return QueryType.PRODUCT_LOOKUP
    

    if retrieval_type == "sparse":
        return QueryType.PRODUCT_LOOKUP
    if retrieval_type == "dense":
        return QueryType.REVIEW_QUERY
    if retrieval_type == "hybrid":
        return QueryType.REVIEW_QUERY
    
    return QueryType.VAGUE
    

def check_phrases(query: str) -> str | None:
    query_lower = query.lower()

    sparse_hit = any(phrase in query_lower for phrase in SPARSE_KEYWORDS if " " in phrase)
    dense_hit = any(phrase in query_lower for phrase in DENSE_KEYWORDS if " " in phrase)
    hybrid_hit = any(word in query_lower for word in HYBRID_KEYWORDS)

    if sparse_hit and dense_hit:
        return "hybrid"
    if hybrid_hit:
        return "hybrid"
    if sparse_hit:
        return "sparse"
    if dense_hit:
        return "dense"
 
    return None
            

def llm_fallback(query:str) -> RouterResult:

    with tracer.start_as_current_span("router_llm_fallback") as span:
        span.set_attribute("router.query", query)
        try:
            ollama_response = f"""
                You are a query classifier for an e-commerce product review assistant.
                Your task is to classify retrieval strategy for user queries into three categories: 
                    1) sparse: for factual queries about price, brand, category, specifications
                    2) dense: for opinion based queries about recommendations, reviews, experiences  
                    3) hybrid: for queries that need both facts and opinions
                            
                Response with json only with the following format:
                    {
                        "retrieval_type": "sparse|dense|hybrid",
                        "reason": "explanation of why this retrieval type was chosen"
                    }
                Classify the intent of following {query} and respond with the appropriate retrieval type and reason.
            """
            
            raw = router_llm.invoke(prompt = ollama_response)
            result = json.loads(raw.content.strip())

            retrieval_type = result.get("retrieval_type")
            query_type = map_query_type(retrieval_type, set(preprocess_query(query)))

            span.set_attribute("llm.output.length", len(str(raw)))
            span.set_attribute("llm.parse.status", "success")

            return RouterResult(
                    retrieval_type=retrieval_type,
                    query_type=query_type,
                    status=RouterStatus.Low_confidence,
                    confidence=0.65,
                    reason=result.get("reason", "LLM classification"),
                    signals={"type": "llm_fallback"},
                    llm_used=True
                )
        
        except Exception as e:
            span.set_attribute("router.status", RouterStatus.Failed.value)
            span.set_attribute("router.failure_reason", str(e))

            return RouterResult(
                retrieval_type="hybrid",
                query_type=QueryType.VAGUE,
                status=RouterStatus.Failed,
                confidence=0.3,
                reason="LLM fallback failed",
                signals={"type": "llm_error"},
                llm_used=True,
                failure_reason=str(e)
            )



def route(query:str) -> RouterResult:
    with tracer.start_as_current_span("router_route") as span:

        span.set_attribute("router.query",query)
        span.set_attribute("router.token_count", len(query.split()))

        tokens = set(preprocess_query(query))
        phrase_result = check_phrases(query)

        if phrase_result:
            return RouterResult(
                retrieval_type=phrase_result,
                query_type=map_query_type(phrase_result, tokens),
                status=RouterStatus.Passed,
                confidence=0.90,
                reason="Phrase match detected",
                signals={"type": "phrase_match"},
                llm_used=False
            )

        if tokens & HYBRID_KEYWORDS:
            matched = tokens & HYBRID_KEYWORDS
            return RouterResult(
                retrieval_type="hybrid",
                query_type=QueryType.COMPARISON,
                status=RouterStatus.Passed,
                confidence=0.80,
                reason=f"Hybrid keywords: {matched}",
                signals={"type": "keyword","matched": list(matched)},
                llm_used=False
            )

        sparse_hits = tokens & SPARSE_KEYWORDS
        dense_hits = tokens & DENSE_KEYWORDS

        if sparse_hits and not dense_hits:
            return RouterResult(
                retrieval_type="sparse",
                query_type=QueryType.PRODUCT_LOOKUP,
                status=RouterStatus.Passed,
                confidence=0.80,
                reason="Sparse keywords",
                signals={"type": "keyword"},
                llm_used=False
            )

        if dense_hits and not sparse_hits:
            return RouterResult(
                retrieval_type="dense",
                query_type=QueryType.REVIEW_QUERY,
                status=RouterStatus.Passed,
                reason="Dense keywords",
                signals={"type": "keyword"},
                llm_used=False,
                confidence=0.80
            )

        if sparse_hits and dense_hits:
            return RouterResult(
                retrieval_type="hybrid",
                query_type=QueryType.REVIEW_QUERY,  
                status=RouterStatus.Ambiguous,
                reason="Conflicting signals",
                signals={"type": "keyword"},
                llm_used=False,
                confidence=0.5
            )

        with tracer.start_as_current_span("router_db_resolution") as db_span:
            db_span.set_attribute("router.query", query)

            match, score, match_type = resolver.resolve("brand", query)

            if match_type == "exact":
                return RouterResult(
                    retrieval_type="sparse",
                    query_type=QueryType.PRODUCT_LOOKUP,
                    status=RouterStatus.Passed,
                    confidence=max(0.80, min(0.90,score)),
                    reason="Exact entity match",
                    signals={"type": "db_exact"},
                    llm_used=False,
                    entity_signal=match
                )

            if match_type == "fuzzy":
                return RouterResult(
                    retrieval_type="sparse",
                    query_type=QueryType.PRODUCT_LOOKUP,
                    status=RouterStatus.Low_Confidence,
                    confidence=0.65,
                    reason="Fuzzy entity match",
                    signals={"type": "db_fuzzy"},
                    llm_used=False,
                    entity_signal=match
                )

        if len(tokens) <= 2:
            return RouterResult(
                retrieval_type="none",
                query_type=QueryType.VAGUE,
                status=RouterStatus.Vague,
                reason="Too vague",
                confidence=0.3,
                signals={"type": "vague"},
                llm_used=False,
                failure_reason="insufficient signal"
            )

        return llm_fallback(query)
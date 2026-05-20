import psycopg2
import os
from dataclasses import dataclass
from typing import Optional
from phoenix_connection import trace_db_query

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "database": os.getenv("POSTGRES_DB", "rag_db"),
    "user": os.getenv("POSTGRES_USER", "rag_user"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT", 5432))
}

ALLOWED_FIELDS = {"asin", "title", "brand"}

class MatchType:
    EXACT = "exact"
    FUZZY = "fuzzy"
    FTS = "fts"
    NONE = "none"


@dataclass
class ResolverResult:
    match_type: str
    entity: Optional[str]
    score: float


class DBEntityLoader:
    def connect(self):
        return psycopg2.connect(**DB_CONFIG)
    
    @trace_db_query
    def exact_match(self, field:str, value:str) -> tuple[str|None, float]:
        if field not in ALLOWED_FIELDS:
            raise ValueError(f"Invalid field: {field}")
        
        query =  f"""
            SELECT {field}
            FROM products_table
            WHERE LOWER({field}) = LOWER(%s)
            LIMIT 1
        """

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query, (value,))
            row = cur.fetchone() 
        
        if row:
            return row[0], 1.0

        return None, 0.0

    @trace_db_query
    def fuzzy_match(self, field:str, value:str) -> tuple[str|None, float]:
        
        if field not in ALLOWED_FIELDS:
            raise ValueError(f"Invalid field: {field}")
        
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

    @trace_db_query
    def full_text_match(self, query_text: str):

        query = """
            SELECT
                title,
                ts_rank(search_vector, plainto_tsquery(%s)) AS rank
            FROM products_table
            WHERE search_vector @@ plainto_tsquery(%s)
            ORDER BY rank DESC
            LIMIT 1
        """

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query, (query_text, query_text))
            row = cur.fetchone()

        if row:
            return row[0], float(row[1])

        return None, 0.0
    
    def resolve(self, query:str):
        try:
            asin_match, score = self.exact_match("asin", query)
            if asin_match:
                return asin_match, score, "asin_exact"
            
            title_match, score = self.exact_match("title", query)
            if title_match:
                return title_match, score, "title_exact"
            
            brand_match, score = self.exact_match("brand", query)
            if brand_match:
                return brand_match, score, "brand_exact"

            title_match, score = self.fuzzy_match("title", query)
            if title_match and score >= 0.75:
                return title_match, score, "title_fuzzy"

            brand_match, score = self.fuzzy_match("brand",query)
            if brand_match and score >= 0.80:
                return brand_match, score, "brand_fuzzy"

            fts_match, score = self.full_text_match(query)
            if fts_match and score >= 0.10:
                return fts_match, score, "fts_match"

            return None, 0.0, "none"

        except Exception:
            return None, 0.0, "error"


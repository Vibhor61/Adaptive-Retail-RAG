import psycopg2
import os
import enum
from dataclasses import dataclass
from typing import Optional, List
from phoenix_connection import trace_db_query

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "database": os.getenv("POSTGRES_DB", "rag_db"),
    "user": os.getenv("POSTGRES_USER", "rag_user"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT", 5432))
}

ALLOWED_FIELDS = {"asin", "title", "brand"}
FUZZY_FIELDS = {"title", "brand"}

class MatchType(enum.Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    FTS_PRODUCT = "fts_product"
    FTS_REVIEW = "fts_review"
    NONE = "none"


@dataclass
class MatchResult:
    match_type: str
    raw_entity: str
    canonical_entity: Optional[str]
    score: float


class DBEntityLoader:
    def connect(self):
        return psycopg2.connect(**DB_CONFIG)
    
    @trace_db_query
    def exact_match(self, field:str, entity:str) -> MatchResult:
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
            cur.execute(query, (entity,))
            row = cur.fetchone() 
        
        if row:
            return MatchResult(
                raw_entity=entity,
                canonical_entity=row[0], 
                score=1.0, 
                match_type=MatchType.EXACT
            )

        conn.close()

        return MatchResult(
            raw_entity=entity,
            canonical_entity=None,
            score=0.0,
            match_type=MatchType.NONE
        )
    

    @trace_db_query
    def fuzzy_match(self, field:str, entity:str) -> MatchResult:
        
        if field not in FUZZY_FIELDS:
            raise ValueError(f"Invalid field: {field}")
        
        query = f"""
            SELECT {field} , similarity(%s, {field}) as score 
            FROM products_table WHERE {field} %% %s
            ORDER BY {field} <-> %s
            LIMIT 1
        """
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query, (entity, entity, entity))
            row = cur.fetchone()
        
        if row:
            return MatchResult(
                raw_entity=entity,
                canonical_entity=row[0],
                score=float(row[1]),
                match_type=MatchType.FUZZY
            )
        
        return MatchResult(
            raw_entity=entity,
            canonical_entity=None,
            score=0.0,
            match_type=MatchType.NONE
        )

    @trace_db_query
    def products_full_text_match(self, query_text: str):

        query = """
            SELECT
                title,
                ts_rank_cd(search_vector, websearch_to_tsquery(%s)) AS rank
            FROM products_table
            WHERE search_vector @@ websearch_to_tsquery(%s)
            ORDER BY rank DESC
            LIMIT 1
        """

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query, (query_text, query_text))
            row = cur.fetchone()

        if row:
            return MatchResult(
                raw_entity=query_text,
                canonical_entity=row[0],
                score=float(row[1]),
                match_type=MatchType.FTS_PRODUCT
            )

        return MatchResult(
            raw_entity=query_text,
            canonical_entity=None,
            score=0.0,
            match_type=MatchType.NONE
        )
    

class EntityResolver:

    def __init__(self, loader: DBEntityLoader):
        self.loader = loader()

    def resolve(self, entities: List[str]) -> list[MatchResult]:
        
        results = []

        for entity in entities:
            best_match = None

            # 1. exact match
            for field in ALLOWED_FIELDS:
                res = self.loader.exact_match(field, entity)
                if res.match_type == MatchType.EXACT:
                    results.append(res)
                    break
            
            # 2. Fuzzy match
            for field in FUZZY_FIELDS:
                res = self.loader.fuzzy_match(field, entity)
                if res.match_type == MatchType.EXACT:
                    results.append(res)
                    break
            
            # 3. fallback FTS
            if best_match is None:
                best_match = self.loader.products_full_text_match(entity)
            results.append(res)

        return results             
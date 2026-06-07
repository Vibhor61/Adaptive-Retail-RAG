import logging

from psycopg2 import pool
from opentelemetry import trace

from contracts.router_contracts import MatchType, CandidateEntity
from config.settings import settings

DB_CONFIG = {
    "host": settings.postgres_host,
    "database": settings.postgres_db,
    "user": settings.postgres_user,
    "password": settings.postgres_password,
    "port": settings.postgres_port
}

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class DBEntityLoader:
    def __init__(self):
        self._pool = pool.SimpleConnectionPool(
            minconn=1, maxconn=10, **DB_CONFIG
        )

    def candidate_search(self, entity: str) -> list[CandidateEntity]:

        word_count = len(entity.split())

        if word_count>3:
            fuzzy_arm = ""
        else:
            fuzzy_arm = "UNION ALL SELECT asin, 'fuzzy' AS match_type FROM products_table WHERE title %% %(ent)s"
        
        query = f"""
            WITH query_parsed AS (
                SELECT websearch_to_tsquery(%(ent)s) AS tsq
            ),
            raw_candidate_ids AS (
                SELECT asin, 'exact' AS match_type FROM products_table WHERE LOWER(title) = LOWER(%(ent)s)
                UNION ALL
                SELECT asin, 'exact' AS match_type FROM products_table WHERE LOWER(brand) = LOWER(%(ent)s)
                UNION ALL
                SELECT asin, 'fts_product' AS match_type FROM products_table, query_parsed WHERE search_vector @@ tsq
                {fuzzy_arm}
            ),
            distinct_pool AS (
                SELECT DISTINCT ON (asin) asin, match_type FROM raw_candidate_ids LIMIT 50
            )
            SELECT
                p.asin,
                p.title,
                p.brand,
                dp.match_type,
                CASE
                    WHEN dp.match_type = 'exact' THEN 1.0
                    WHEN dp.match_type = 'fuzzy' THEN similarity(p.title, %(ent)s)
                    ELSE LEAST(1.0, ts_rank_cd(p.search_vector, (SELECT tsq FROM query_parsed)))
                END AS score
            FROM distinct_pool dp
            JOIN products_table p ON dp.asin = p.asin
            ORDER BY score DESC
            LIMIT 20;
        """

        candidates: list[CandidateEntity] = []

        with tracer.start_as_current_span("router.db_candidate_search") as span:
            span.set_attribute("entity_or_query", entity)
            
            try:
                conn = self._pool.getconn()
                with conn.cursor() as cur:
                    # If autocommit=True threshold will not apply 
                    cur.execute("SET LOCAL pg_trgm.similarity_threshold = 0.40")
                    cur.execute(query, {"ent": entity})
                    rows = cur.fetchall()

                span.set_attribute("row_count", len(rows))

                for row in rows:
                    candidates.append(CandidateEntity(
                        asin=row[0] or None,
                        title=row[1] or "unknown",
                        brand=row[2] or None,
                        match_type=MatchType(row[3] or "none"),
                        retrieval_score=float(row[4] or 0.0),
                    ))

                return candidates

            except Exception as e:
                logger.error("DBEntityLoader candidate search failed for %r : %s", entity, e)
                span.record_exception(e)
                raise
            finally:
                self._pool.putconn(conn)
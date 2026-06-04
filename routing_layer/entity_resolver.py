import logging
import psycopg2

from typing import List
from opentelemetry import trace
from config.settings import settings
from contracts.router_contracts import (
    MatchType,
    GroundedEntity
)

DB_CONFIG = {
    "host": settings.postgres_host,
    "database": settings.postgres_db,
    "user": settings.postgres_user,
    "password": settings.postgres_password,
    "port": settings.postgres_port
}

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

MIN_FUZZY_SCORE = 0.40


class DBEntityLoader:
    def connect(self):
        return psycopg2.connect(**DB_CONFIG)

    def candidate_search(self, entity: str):
        query = """
            SELECT asin, title, brand,             

            CASE 
                WHEN LOWER(title) = LOWER(%s) THEN 'exact'
                WHEN LOWER(brand) = LOWER(%s) THEN 'exact'
                WHEN title %% %s AND similarity(title,%s) > 0.40 THEN 'fuzzy'
                WHEN search_vector @@ websearch_to_tsquery(%s) THEN 'fts_product'
                ELSE 'none'
            END AS match_type,

            CASE 
                WHEN LOWER(title) = LOWER(%s) THEN 1.0
                WHEN LOWER(brand) = LOWER(%s) THEN 1.0
                WHEN title %% %s AND similarity(title, %s) > 0.40 THEN similarity(%s, title)
                ELSE ts_rank_cd(search_vector, websearch_to_tsquery(%s))
            END AS score

            FROM products_table
            WHERE 
                LOWER(title) = LOWER(%s)
            OR LOWER(brand) = LOWER(%s)
            OR title %% %s
            OR search_vector @@ websearch_to_tsquery(%s)

        LIMIT 20;
        """

        with tracer.start_as_current_span("router.db.candidate_search") as span:
            span.set_attribute("entity", entity)
            try:
                with self.connect() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        query,
                        (entity, entity, entity, entity, entity, entity, entity, entity, entity, entity, entity, entity, entity, entity, entity)
                    )
                    rows = cur.fetchall()
                    span.set_attribute("row_count", len(rows))
                    return rows
            except Exception as e:
                logger.error("DBEntityLoader candidate search failed for %r : %s", entity, e)
                span.record_exception(e)
                raise


class EntityResolver:

    def __init__(self, loader: DBEntityLoader):
        self.loader = loader

    def _rank(self, entity: str, rows) -> GroundedEntity:

        with tracer.start_as_current_span("router.entity.rank") as span:

            if not rows:
                span.set_attribute("result", "no_candidates")
                return GroundedEntity(
                    raw_entity=entity,
                    canonical_entity=None,
                    match_type=MatchType.NONE,
                    score=0.0
                )

            best_score = -1.0
            second_best = -1.0
            best = None

            type_weight = {
                MatchType.EXACT: 1.0,
                MatchType.FUZZY: 0.7,
                MatchType.FTS_PRODUCT: 0.5,
                MatchType.NONE: 0.0
            }

            for r in rows:
                asin, title, brand, match_type, score = r
                mt = MatchType(match_type)

                final_score = (
                    0.75 * float(score or 0.0)
                    + 0.25 * type_weight[mt]
                )

                if final_score > best_score:
                    second_best = best_score
                    best_score = final_score
                    best = (asin, title, brand, mt, final_score)

            asin, title, brand, mt, final_score = best

            span.set_attribute("winner.asin", asin)
            span.set_attribute("winner.match_type", mt.value)
            span.set_attribute("winner.score", final_score)
            span.set_attribute("decision.margin", best_score - second_best)
            
            return GroundedEntity(
                raw_entity=entity,
                canonical_entity=title,
                match_type=mt,
                score=final_score
            )

    def resolve(self, entities: List[str]) -> list[GroundedEntity]:
        results = []

        with tracer.start_as_current_span("router.db") as span:
            for entity in entities:
                try:
                    rows = self.loader.candidate_search(entity)
                    grounded = self._rank(entity, rows)
                    results.append(grounded)
                except Exception as e:
                    span.record_exception(e)
                    logger.error("EntityResolver.resolve failed for entity=%r: %s", entity, e)
                    raise

        return results
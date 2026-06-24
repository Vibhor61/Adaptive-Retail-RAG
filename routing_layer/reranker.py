import logging
import torch

from opentelemetry import trace

from contracts.router_contracts import (
    Intent,
    RankedCandidate,
    CandidateEntity,
    EntityStructure,
)

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class EntityReranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, candidates: list[CandidateEntity],) -> list[RankedCandidate]:

        with tracer.start_as_current_span("reranker.rerank") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_candidates", len(candidates))

            if not candidates:
                return []

            pairs = [(query, c.title or "") for c in candidates]

            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            ranked: list[RankedCandidate] = []

            for c, score in zip(candidates, scores):
                ranked.append(
                    RankedCandidate(
                        asin=c.asin,
                        title=c.title,
                        brand=c.brand,
                        retrieval_score=float(c.retrieval_score),
                        reranker_score=float(torch.sigmoid(torch.tensor(score))),
                    )
                )

            ranked.sort(key=lambda x: x.reranker_score, reverse=True)

            if ranked:
                span.set_attribute("top.title", ranked[0].title or "none")
                span.set_attribute("top.asin", ranked[0].asin or "none")
                span.set_attribute("top.retrieval_score", round(ranked[0].retrieval_score, 4))
                span.set_attribute("top.reranker_score", round(ranked[0].reranker_score, 4))

            if len(ranked) > 1:
                span.set_attribute("margin", round(ranked[0].reranker_score - ranked[1].reranker_score, 4))

            span.set_attribute(
                "candidate_titles", str([c.title for c in candidates[:5]])
            )

            return ranked


class EntityResolver:
    def __init__(self, loader, reranker: EntityReranker):
        self.loader = loader
        self.reranker = reranker

    def resolve(
        self, query: str, intent: Intent, entities: list[str], entity_structure: EntityStructure,
    ) -> list[RankedCandidate]:

        with tracer.start_as_current_span("resolver.resolve") as span:
            span.set_attribute("query", query)
            span.set_attribute("intent", intent.value)
            span.set_attribute("entity_structure", entity_structure.value)
            span.set_attribute("entities", str(entities))
            span.set_attribute("num_entities", len(entities))

            if intent in (Intent.RECOMMENDATION, Intent.UNKNOWN):
                span.set_attribute("skipped", True)
                span.set_attribute("skip_reason", intent.value)
                return [], []

            try:
                if not entities:
                    span.set_attribute("fallback_used", True)
                    # no entities from LLM — run full query as fallback
                    candidates = self.loader.candidate_search(query)
                    if not candidates:
                        return [], []
                    ranked = self.reranker.rerank(query=query, candidates=candidates)
                    if not ranked:
                        return [], candidates
                    
                    span.set_attribute("fallback_top_title",ranked[0].title or "none")
                    span.set_attribute("fallback_top_score", ranked[0].reranker_score)

                    return [ranked[0]], candidates
        
                else:
                    span.set_attribute("fallback_used", False)

                grounded: list[RankedCandidate] = []
                all_candidates: list[CandidateEntity] = []

                for entity in entities:
                    with tracer.start_as_current_span("resolver.resolve.entity") as entity_span:
                        entity_span.set_attribute("entity", entity)

                        candidates = self.loader.candidate_search(entity)
                        entity_span.set_attribute("num_candidates", len(candidates))
                        all_candidates.extend(candidates)
                        if not candidates:
                            logger.warning("No candidates found for entity=%r", entity)
                            entity_span.set_attribute("found", False)
                            continue

                        ranked = self.reranker.rerank(query=query, candidates=candidates)

                        if not ranked:
                            continue

                        entity_span.set_attribute("found", True)
                        entity_span.set_attribute("top.title", ranked[0].title or "none")
                        entity_span.set_attribute("top.reranker_score", round(ranked[0].reranker_score, 4))
                        entity_span.set_attribute("top.asin", ranked[0].asin or "none")
                        entity_span.set_attribute("top.retrieval_score", round(ranked[0].retrieval_score, 4))   
                        
                        grounded.append((ranked[0]))

                span.set_attribute("num_grounded", len(grounded))
                span.set_attribute("grounded_asins", str([c.asin for c in grounded]))
                return grounded, all_candidates

            except Exception as e:
                span.record_exception(e)
                logger.error("EntityResolver failed for query=%r: %s", query, e)
                raise
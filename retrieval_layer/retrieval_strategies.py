from contracts.router_contracts import (
    EvidenceType,
    EntityStructure,
)

from contracts.retrieval_contracts import (
    RetrievalPlan,
    RetrievalEvaluationBundle,
)

from contracts.orchestration_contracts import (
    RetrievalLayerOutput
)

from retrieval_layer.retrievers import (
    sparse_fact_retrieval,
    fusion_retrieval,
    candidate_gen_retrieval,
)

from retrieval_layer.retrieval_evaluation import evaluate_retrieval


def sparse_for_entity(entity, top_k: int) -> RetrievalEvaluationBundle:
    bundle = sparse_fact_retrieval(
        entity=entity.canonical_entity,
        top_k=top_k,
    )
    return evaluate_retrieval(bundle)


def fusion_for_entity(entity, query: str, top_k: int) -> RetrievalEvaluationBundle:
    bundle = fusion_retrieval(
        query=entity.canonical_entity or query,
        top_k=top_k,
    )
    return evaluate_retrieval(bundle)


def sparse_and_fusion_for_entity(entity, query: str, top_k: int) -> list[RetrievalEvaluationBundle]:
    return [
        sparse_for_entity(entity, top_k),
        fusion_for_entity(entity, query, top_k),
    ]


def by_evidence(plan: RetrievalPlan, entity) -> list[RetrievalEvaluationBundle]:
    """Dispatch to the right primitive(s) based on evidence type for a single entity."""
    if plan.evidence_type == EvidenceType.FACTUAL:
        return [sparse_for_entity(entity, plan.top_k)]

    elif plan.evidence_type == EvidenceType.EXPERIENTIAL:
        return [fusion_for_entity(entity, plan.original_query, plan.top_k)]

    elif plan.evidence_type == EvidenceType.MIXED:
        return sparse_and_fusion_for_entity(entity, plan.original_query, plan.top_k)

    raise NotImplementedError(
        f"evidence_type '{plan.evidence_type}' not handled — adaptive routing required."
    )


def lookup_workflow(plan: RetrievalPlan) -> RetrievalLayerOutput:
    evaluation_bundles: list[RetrievalEvaluationBundle] = []

    if plan.entity_structure in (EntityStructure.SINGLE, EntityStructure.MULTI_EXPLICIT):
        for entity in plan.grounded_entities:
            evaluation_bundles.extend(by_evidence(plan, entity))

    elif plan.entity_structure in (EntityStructure.MULTI_IMPLICIT, EntityStructure.NONE):
        # No grounded entities — fall back to raw query fusion
        bundle = fusion_retrieval(query=plan.original_query, top_k=plan.top_k)
        evaluation_bundles.append(evaluate_retrieval(bundle))

    else:
        raise NotImplementedError(
            f"entity_structure '{plan.entity_structure}' not handled in lookup — adaptive routing required."
        )

    return RetrievalLayerOutput(plan=plan, evaluation_bundles=evaluation_bundles)



def comparison_workflow(plan: RetrievalPlan) -> RetrievalLayerOutput:

    evaluation_bundles: list[RetrievalEvaluationBundle] = []

    if plan.entity_structure == EntityStructure.MULTI_EXPLICIT:
        for entity in plan.grounded_entities:
            evaluation_bundles.extend(by_evidence(plan, entity))

    else:
        raise NotImplementedError(
            f"comparison with entity_structure '{plan.entity_structure}' "
            "is ambiguous — adaptive routing required."
        )

    return RetrievalLayerOutput(plan=plan, evaluation_bundles=evaluation_bundles)


def recommendation_workflow(plan: RetrievalPlan) -> RetrievalLayerOutput:
  
    evaluation_bundles: list[RetrievalEvaluationBundle] = []

    if plan.entity_structure in (EntityStructure.MULTI_IMPLICIT, EntityStructure.NONE):
        bundle = candidate_gen_retrieval(query=plan.original_query, top_k=plan.top_k)
        evaluation_bundles.append(evaluate_retrieval(bundle))

    elif plan.entity_structure == EntityStructure.SINGLE:
        entity = plan.grounded_entities[0]
        evaluation_bundles.extend(
            sparse_and_fusion_for_entity(entity, plan.original_query, plan.top_k)
        )

    else:
        raise NotImplementedError(
            f"recommendation with entity_structure '{plan.entity_structure}' "
            "not handled — adaptive routing required."
        )

    return RetrievalLayerOutput(plan=plan, evaluation_bundles=evaluation_bundles)
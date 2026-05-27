import enum

from dataclasses import dataclass
from typing import List
from opentelemetry import trace
from langchain_ollama import OllamaLLM

from utils import safe_llm_call
from pydantic import BaseModel, Field


tracer = trace.get_tracer(__name__)

router_llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0)

class QueryType(enum.Enum):
    PRODUCT_LOOKUP = "product_lookup"
    PRODUCT_ATTRIBUTE_QUERY = "product_attribute_query"
    REVIEW_QUERY = "review_query"
    COMPARISON = "comparison"
    MULTI_ENTITY_REVIEW = "multi_entity_review"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class EntityStructure(enum.Enum):
    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"


class AnswerType(enum.Enum):
    LOOKUP = "lookup"
    FACTUAL = "factual"
    EXPERIENTIAL = "experiential"
    COMPARATIVE = "comparative"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


class IntentNature(enum.Enum):
    FACTUAL = "factual"
    SUBJECTIVE = "subjective"
    EXPLORATORY = "exploratory"


class RetrievalCapability(enum.Enum):
    PRODUCT_METADATA = "product_metadata"
    REVIEW_SPARSE = "review_sparse"
    REVIEW_DENSE = "review_dense"


class ExtractedEntity(BaseModel):
    test: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class RouterOutput(BaseModel):
   
    query_type: QueryType

    entities: List[ExtractedEntity] = Field(
        default_factory=list
    )

    entity_structure: EntityStructure

    answer_type: AnswerType

    intent_nature: IntentNature

    retrieval_capabilities: List[RetrievalCapability] = Field(
        default_factory=list
    )

    ambiguity_detected: bool = False

    requires_clarification: bool = False

    routing_signals: List[str] = Field(
        default_factory=list
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )



def analyze_intent(query: str) -> IntentResult:

    with tracer.start_as_current_span("intent_router") as span:
        
        span.set_attribute("intent.query", query)
        
        prompt = f"""
        You are a query routing system for a retail RAG platform.

        Your job:
        1. classify query intent
        2. extract raw product entities

        Allowed query types:

        - product_lookup
        - product_attribute_query
        - review_query
        - comparison
        - multi_entity_review
        - hybrid
        - unknown

        Definitions:

        product_lookup:
        - direct product search
        - entity-focused
        Examples:
        - iphone 15
        - sony xm5

        product_attribute_query:
        - factual product questions
        Examples:
        - what is battery life of iphone 15
        - price of sony xm5

        review_query:
        - recommendation/review queries without explicit entities
        Examples:
        - best headphones for gym
        - comfortable office chair

        comparison:
        - explicit comparison intent
        Examples:
        - iphone 15 vs samsung s24
        - compare bose and sony headphones

        multi_entity_review:
        - experiential queries involving multiple entities
        Examples:
        - how do sony xm5 and bose qc ultra perform for flights

        hybrid:
        - single entity + experiential/review intent
        Examples:
        - is sony xm5 good for gaming
        - are airpods worth it

        unknown:
        - unclear or ambiguous intent

        Return ONLY valid JSON.

        Schema:
        {
        "query_type": "...",
        "entities": [...],
        "confidence": 0.0
        }
        """

        try:
            parsed = safe_llm_call(router_llm, prompt, "json")

            validated = RouterOutput(**parsed)
            
            span.set_attribute("router.query_type", validated.query_type.value)
            span.set_attribute("router.query_entities", str(validated.entities))
            span.set_attribute("router.confidence", validated.confidence)
            span.set_attribute("router.status", "success")


            return IntentResult(
                query_type=validated.query_type,
                entities=validated.entities,
                confidence=validated.confidence,
                raw_response=parsed
            )

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("intent.error", str(e))
            span.set_attribute("router.status", "success")

            return IntentResult(
                query_type=QueryType.UNKNOWN,
                entities=[],
                confidence=0.0,
                raw_response={}
            )
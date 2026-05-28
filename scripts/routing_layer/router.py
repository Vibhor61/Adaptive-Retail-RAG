import enum

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
    MULTI_PRODUCT_REASONING = "multi_product_reasoning"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


class EntityStructure(enum.Enum):
    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"


class AnswerType(enum.Enum):
    FACTUAL = "factual"
    EXPERIENTIAL = "experiential"
    COMPARATIVE = "comparative"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


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

    anomaly_signals: List[str] = Field(
        default_factory=list
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )



def analyze_intent(query: str) -> RouterOutput:

    with tracer.start_as_current_span("intent_router") as span:
        
        span.set_attribute("intent.query", query)
        
        prompt = f"""
        You are a semantic router for a retail RAG system.
        Your task is ONLY semantic interpretation.

        You must:

        1. classify query type
        2. extract raw product entities
        3. determine entity structure
        4. determine answer type
        5. estimate confidence

        --------------------------------------------------
        Allowed query types
        --------------------------------------------------

        product_lookup
        - direct product search
        - exact product lookup
        Examples:
        - iphone 15
        - sony xm5
        - samsung s24 ultra

        product_attribute_query
        - factual product information
        - specifications or metadata
        Examples:
        - price of iphone 15
        - battery life of sony xm5
        - what brand is this monitor

        review_query
        - experiential or opinion-oriented query
        - focused on user experience/reviews
        Examples:
        - is sony xm5 comfortable
        - does this mouse double click
        - how is camera quality at night

        comparison
        - explicit comparison between products/entities
        Examples:
        - iphone 15 vs s24
        - compare bose qc ultra and sony xm5

        multi_product_reasoning
        - reasoning/recommendation across multiple candidate products
        - exploratory shopping intent
        Examples:
        - best gaming headphones
        - good laptops for programming
        - best monitors under 300

        recommendation
        - direct recommendation seeking
        Examples:
        - recommend earbuds for gym
        - suggest a good office chair

        unknown
        - ambiguous or unclear intent

        --------------------------------------------------
        Allowed entity structures
        --------------------------------------------------

        none
        - no clear product entities

        single
        - one product entity

        multi
        - multiple product entities

        --------------------------------------------------
        Allowed answer types
        --------------------------------------------------

        factual
        - objective metadata/specifications

        experiential
        - review/opinion/user experience

        comparative
        - comparison between entities

        recommendation
        - exploratory recommendation guidance

        unknown
        - unclear answer expectation

        --------------------------------------------------
        Rules
        --------------------------------------------------

        - Return ONLY valid JSON
        - Do not include explanations
        - Do not invent entities
        - Extract raw entity text exactly from query
        - Confidence must be between 0.0 and 1.0

        --------------------------------------------------
        Output Schema
        --------------------------------------------------

        {{
        "query_type": "...",
        "entities": [
            {{
            "text": "...",
            "confidence": 0.0
            }}
        ],
        "entity_structure": "...",
        "answer_type": "...",
        "anomaly_signals": [],
        "confidence": 0.0
        }}

        --------------------------------------------------
        User Query
        --------------------------------------------------

        {query}
        """


        try:
            parsed = safe_llm_call(router_llm, prompt, "json")

            validated = RouterOutput(**parsed)
            
            span.set_attribute("router.query_type", validated.query_type.value)
            span.set_attribute("router.query_entities", str(validated.entities))
            span.set_attribute("router.confidence", validated.confidence)
            span.set_attribute("router.status", "success")

            return validated

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("intent.error", str(e))
            span.set_attribute("router.status", "error")

            return RouterOutput(
                query_type=QueryType.UNKNOWN,
                entities=[],
                entity_structure=EntityStructure.NONE,
                answer_type=AnswerType.UNKNOWN,
                anomaly_signals=[str(e)],
                confidence=0.0
            )
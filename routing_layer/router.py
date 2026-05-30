from typing import List
from opentelemetry import trace
from langchain_ollama import OllamaLLM

from utils import safe_llm_call
from contracts.router_contracts import (
    RouterOutput,
    Intent,
    EntityStructure,
    EvidenceType
)


tracer = trace.get_tracer(__name__)

router_llm = OllamaLLM(model="qwen2.5:7b", temperature=0)


def analyze_intent(query: str) -> RouterOutput:

    with tracer.start_as_current_span("intent_router") as span:
        
        span.set_attribute("intent.query", query)
        
        prompt = f"""
        You are a semantic router for a retail RAG system.
        Your responsibility is query understanding only.

        You must:

        1. classify intent
        2. extract product entities
        3. determine entity structure
        4. determine evidence type
        5. extract user constraints
        6. estimate confidence

        --------------------------------------------------
        Intent Types
        --------------------------------------------------

        lookup

        - user seeks information about products
        - includes factual questions
        - includes review/opinion questions

        Examples:

        - iphone 15
        - price of iphone 15
        - what is iphone 15 battery capacity
        - does iphone 15 overheat
        - how is sony xm5 comfort
        - tell me about galaxy s24

        comparison

        - user compares products

        Examples:

        - iphone 15 vs galaxy s24
        - compare sony xm5 and bose qc ultra
        - which phone has better battery life

        recommendation

        - user seeks suggestions
        - user seeks alternatives
        - user seeks best products for a purpose

        Examples:

        - best gaming laptop
        - recommend earbuds for gym
        - alternatives to iphone 15
        - best phones under 50000

        unknown

        - intent unclear

        --------------------------------------------------
        Entity Structure
        --------------------------------------------------

        none

        - no identifiable product entities

        Examples:

        - what laptop should I buy

        single

        - one explicit product entity

        Examples:

        - iphone 15 battery life

        multi_explicit

        - multiple explicit product entities

        Examples:

        - iphone 15 vs galaxy s24
        - compare sony xm5 and bose qc ultra

        multi_implicit

        - user is asking for multiple candidate products
        - products are not explicitly named

        Examples:

        - best gaming laptops
        - recommend phones under 50000
        - top wireless earbuds

        --------------------------------------------------
        Evidence Types
        --------------------------------------------------

        factual

        - objective product facts
        - specifications
        - metadata

        Examples:

        - battery capacity
        - display size
        - processor
        - weight
        - price

        experiential

        - user experiences
        - reviews
        - opinions
        - satisfaction

        Examples:

        - does it overheat
        - are users satisfied
        - how comfortable is it
        - durability issues

        mixed

        - requires both factual and experiential evidence

        Examples:

        - is it good for gaming
        - is it worth buying
        - tell me about iphone 15
        - compare iphone 15 and galaxy s24

        --------------------------------------------------
        Constraint Extraction
        --------------------------------------------------

        Extract any user constraints.

        Examples:

        best laptop under 100000

        {{
        "price_upper": 100000
        }}

        gaming laptop

        {{
        "use_case": "gaming"
        }}

        wireless earbuds

        {{
        "category": "wireless earbuds"
        }}

        Return constraints inside:

        {{
        "raw_constraints": {{}}
        }}

        --------------------------------------------------
        Rules
        --------------------------------------------------

        - return only valid JSON
        - do not explain reasoning
        - do not invent entities
        - extract entities exactly as written
        - confidence must be between 0.0 and 1.0
        - anomaly_signals must be a list of strings

        --------------------------------------------------
        Output Schema
        --------------------------------------------------

        {{
        "intent": "...",
        "entities": [
            {{
            "text": "...",
            "confidence": 0.0
            }}
        ],
        "entity_structure": "...",
        "evidence_type": "...",
        "constraints": {{
            "raw_constraints": {{}}
        }},
        "confidence": 0.0
        }}

        --------------------------------------------------
        User Query
        --------------------------------------------------

        {query}
        """


        try:
            parsed = safe_llm_call(router_llm, prompt, "json")
            span.set_attribute("router.raw_response",str(parsed))

            validated = RouterOutput(**parsed)
            
            span.set_attribute("router.intent_type", validated.intent_type.value)
            span.set_attribute("router.query_entities", str(validated.entities))
            span.set_attribute("router.confidence", validated.confidence)
            span.set_attribute("router.status", "success")

            return validated

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("intent.error", str(e))
            span.set_attribute("router.status", "error")

            return RouterOutput(
                intent_type=Intent.UNKNOWN,
                entities=[],
                entity_structure=EntityStructure.NONE,
                evidence_type=EvidenceType.MIXED,
                constraints=None,
                confidence=0.0
            )
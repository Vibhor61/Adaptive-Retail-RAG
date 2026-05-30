from typing import List, Optional, Tuple
import re
from langchain_ollama import OllamaLLM
from dataclasses import dataclass
import enum
from opentelemetry import trace
from langchain_ollama import OllamaLLM 

tracer = trace.get_tracer(__name__)


def build_prompt(query:str, postgres_results: List[RetrievalResult]|None, qdrant_results: List[RetrievalResult]|None) -> str:
    prompt =f"""  
        You are an assistant of e-commerce product review analysis.
        Your task is to analyze user queries to provide accurate and helpful answers based on retrieved product or review information.
        You will be provided with a user query and relevant information from product database(sparse retrieval) or customer reviews (dense retrieval) or both.

        User Query: {query}
    """

    if postgres_results:
        prompt += "\n [PRODUCT FACTS] (from sparse retrieval):\n"
        for item in postgres_results:
            title = item.metadata.get("title", "Unknown")
            brand = item.metadata.get("brand", "Unknown")
            category = item.metadata.get("category", "Unknown")
            price = item.metadata.get("price", "Unknown")
            price_raw = item.metadata.get("price_raw", "Unknown")
            prompt += f" Product: {title} | Brand: {brand} | Category: {category} | Price: {price} ({price_raw}) | Score: {item.score:.2f}\n"

    if qdrant_results:
        prompt += "\n [CUSTOMER REVIEWS] (from dense retrieval):\n"
        for item in qdrant_results:
            asin = item.metadata.get("asin")
            if not asin:
                continue
            prompt += f"- Product {asin}: {item.text} | Relevance: {item.score:.2f}\n"

    prompt += """
        [INSTRUCTIONS]
        - Answer concisely and directly
        - Cite whether your answer comes from product facts or customer reviews
        - If asked for opinion, use customer reviews
        - If asked for specs or price, use product facts
        - Based on the information provided, answer the user's query. If information is insufficient to provide confident answer, say you don't know.
        - Do not make up information not present above
    """

    return prompt




def answer_query(query: str, retrieval_result: FinalResult, level: int = 1) -> GenerationResult:
    with tracer.start_as_current_span("answer_query") as span:
        if not retrieval_result or not retrieval_result.items:
            span.set_attribute("answer_query.status", "no_retrieval_results")
            return GenerationResult(
                answer="",
                model_used="",
                score=0.0,
                status=GenerationStatus.EMPTY,
                failure_type="no_retrieval_results",
                failure_reason="No retrieval results provided",
                signals=None
            )

        postgres_results = [item for item in retrieval_result.items if item.source in ("sparse", "postgres")]
        qdrant_results = [item for item in retrieval_result.items if item.source in ("dense", "qdrant")]

        span.set_attribute("answer_query.postgres_count", len(postgres_results))
        span.set_attribute("answer_query.qdrant_count", len(qdrant_results))

        prompt = build_prompt(query, postgres_results, qdrant_results)
        answer_text, model_name = generate_answer(prompt, level)

        status, score, failure_reason, signals = validate_answer(answer_text, query, postgres_results, qdrant_results)

        span.set_attribute("answer_query.model", model_name)
        span.set_attribute("answer_query.status", status.value)
        span.set_attribute("answer_query.score", score)
        if failure_reason:
            span.set_attribute("answer_query.failure_reason", failure_reason)

        return GenerationResult(
            answer=answer_text,
            model_used=model_name,
            score=score,
            status=status,
            failure_reason=failure_reason,
            signals=signals
        )
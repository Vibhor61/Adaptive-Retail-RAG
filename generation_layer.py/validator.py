import re

from opentelemetry import trace
from typing import List, Tuple

from contracts.retrieval_contracts import RetrievalResult
from generation_contracts import GenerationStatus, ValidationSignals, GenerationResult
from models import get_model

tracer = trace.get_tracer(__name__)

def classify_generation_exceptions(exception: Exception) -> str:
    exception_str = str(exception).lower()

    if "429" in exception_str or "rate limit" in exception_str:
        return "rate_limit"
    
    if "timeout" in exception_str or "timed out" in exception_str:
        return "timeout"
    
    if "503" in exception_str or "service unavailable" in exception_str or "connection" in exception_str:
        return "service_unavailable"
    
    if "context" in exception_str or "length" in exception_str or "token" in exception_str:
        return "context_overflow"
    
    if "parse" in exception_str or "json" in exception_str:
        return "response_parsing_error"
    
    if "401" in exception_str or "unauthorized" in exception_str or "api key" in exception_str:
        return "authentication_error"
    
    return "unknown_generation_exception"


def validate_answer(answer:str, query:str, postgres_results: List[RetrievalResult]|None, qdrant_results: List[RetrievalResult]|None) -> tuple[GenerationStatus, float, str, ValidationSignals]:
    
    with tracer.start_as_current_span("answer_validation") as span:
        
        normalized_answer = answer.lower()
        has_product_facts = len(postgres_results) > 0 if postgres_results else False
        has_customer_reviews = len(qdrant_results) > 0 if qdrant_results else False
        
        span.set_attribute("validation.has_product_facts", has_product_facts)
        span.set_attribute("validation.has_customer_reviews", has_customer_reviews)
        span.set_attribute("validation.answer_length", len(answer))

        product_titles = set()
        sparse_asins = set()
        product_brands = set()
        product_categories = set()
        
        if has_product_facts:
            for item in postgres_results:
                metadata = item.metadata or {}
                title = metadata.get("title", "").lower()
                brand = metadata.get("brand", "").lower()
                category = metadata.get("category", "").lower()
                asin = metadata.get("asin", "").lower()
                if title:
                    product_titles.add(title)
                if brand:
                    product_brands.add(brand)
                if category:
                    product_categories.add(category)
                if not asin:
                    continue
                sparse_asins.add(asin)

        dense_asins = set()
        if has_customer_reviews:
            for item in qdrant_results:
                metadata = item.metadata or {}
                asin = metadata.get("asin", "").lower()
                if not asin:
                    continue
                dense_asins.add(asin)

        cited_product_facts = any(title in normalized_answer for title in product_titles) or \
            any(brand in normalized_answer for brand in product_brands) or \
            any(category in normalized_answer for category in product_categories) or \
            any(asin in normalized_answer for asin in sparse_asins)

        cited_customer_reviews = any(asin in normalized_answer for asin in dense_asins)
        review_phrases = [
            "reviews mention",
            "reviewers mention",
            "customers say",
            "customer reviews",
            "buyers mention",
            "users report"
        ]
        cited_customer_reviews = cited_customer_reviews or any(phrase in normalized_answer for phrase in review_phrases)

        query_tokens = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                query.lower()
            )
        )

        answer_tokens = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                normalized_answer
            )
        )

        overlap = query_tokens.intersection(answer_tokens)

        coverage_score = (
            len(overlap) / len(query_tokens)
            if query_tokens else 1.0
        )

        signals = ValidationSignals(
            has_product_facts=has_product_facts,
            has_customer_reviews=has_customer_reviews,
            answer_length=len(answer),
            cited_product_facts=cited_product_facts,
            cited_customer_reviews=cited_customer_reviews,
            mentioned_asins=len(sparse_asins.union(dense_asins)),
            mentioned_titles=len(product_titles),
            coverage_score=coverage_score
        )

        if not normalized_answer:
            span.set_attribute("validation.status", "empty")
            return (
                GenerationStatus.EMPTY,
                0.0,
                "Empty answer",
                signals
            )

        if len(normalized_answer) < 25:
            span.set_attribute("validation.status", "too_short")
            return (
                GenerationStatus.TOO_SHORT,
                0.2,
                "Answer too short",
                signals
            )

        refusal_patterns = [
            "i don't know",
            "not enough information",
            "cannot determine",
            "insufficient information"
        ]

        if any(
            pattern in normalized_answer
            for pattern in refusal_patterns
        ):
            span.set_attribute("validation.status", "refusal")
            return (
                GenerationStatus.REFUSAL,
                0.3,
                "Model refusal",
                signals
            )

        if cited_product_facts and not has_product_facts:
            span.set_attribute("validation.status", "attribution_error")
            return (
                GenerationStatus.ATTRIBUTION_ERROR,
                0.5,
                "Claims product facts without sparse retrieval",
                signals
            )

        if cited_customer_reviews and not has_customer_reviews:
            span.set_attribute("validation.status", "attribution_error")
            return (
                GenerationStatus.ATTRIBUTION_ERROR,
                0.5,
                "Claims customer reviews without dense retrieval",
                signals
            )

        if coverage_score < 0.20:
            span.set_attribute("validation.status", "low_coverage")
            return (
                GenerationStatus.LOW_COVERAGE,
                0.5,
                "Poor query coverage",
                signals
            )

        span.set_attribute("validation.status", "passed")
        return (
            GenerationStatus.PASSED,
            1.0,
            "Generation passed validation",
            signals
        )


def generate_answer(prompt:str, model_number:int = 1) -> Tuple[str, str]:
    llm, model_name = get_model(model_number)

    with tracer.start_as_current_span("answer_generation") as span:
        span.set_attribute("answer.model", model_name)
        span.set_attribute("answer.level", model_number)

        try: 
            response = llm.invoke(prompt)
            if isinstance(response, str):
                answer_text = response.strip()
            else:
                answer_text = response.content.strip()

            if hasattr(response, "usage"):
                span.set_attribute("llm.total_tokens", getattr(response.usage, "total_tokens", 0))

            span.set_attribute("answer.length", len(answer_text))
            span.set_attribute("answer.status", "success")

            return answer_text, model_name

        except Exception as e:
            span.set_attribute("answer.status", "error")
            span.set_attribute("answer.failure_reason", str(e))
            
            return GenerationResult(
                answer="",
                model_used=model_name,
                score=0.0,
                status=GenerationStatus.EXCEPTION,
                failure_reason=classify_generation_exceptions(e),
                signals=None
            )
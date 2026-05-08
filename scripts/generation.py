from typing import List, Optional, Tuple
import re
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from retrieval import RetrievalResult, FinalResult
from dataclasses import dataclass
import enum
from opentelemetry import trace

llm_ollama = OllamaLLM(model = "mistral:latest")
groq_llm = ChatGroq(model="llama3-8b-8192")
gemini_flash_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
gemini_pro_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

tracer = trace.get_tracer(__name__)

class GenerationStatus(enum.Enum):
    PASSED = "passed"
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    REFUSAL = "refusal"
    LOW_COVERAGE = "low_coverage"
    ATTRIBUTION_ERROR = "attribution_error"
    EXCEPTION = "exception"


@dataclass
class ValidationSignals:
    has_product_facts: bool
    has_customer_reviews: bool
    answer_length: int
    cited_product_facts: bool
    cited_customer_reviews: bool
    mentioned_asins: int
    mentioned_titles: int
    coverage_score: Optional[float]


@dataclass
class GenerationResult:
    answer: str
    model_used: str

    score: Optional[float]
    status: Optional[GenerationStatus]
    failure_reason: Optional[str]
    signals: Optional[ValidationSignals]


def select_model(model_number:int):
    if model_number == 1:
        return groq_llm, "llama3-8b-8192"
    elif model_number == 2:
        return gemini_flash_llm, "gemini-2.0-flash"
    else:
        return gemini_flash_llm, "gemini-1.5-pro"
    

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
    llm, model_name = select_model(model_number)

    with tracer.start_as_current_span("answer_generation") as span:
        span.set_attribute("answer.model", model_name)
        span.set_attribute("answer.level", model_number)

        try: 
            response = llm.invoke(prompt)
            answer_text = response.content.strip()

            if hasattr(response, "usage"):
                span.set_attribute("llm.total_tokens", getattr(response.usage, "total_tokens", 0))

            span.set_attribute("answer.length", len(answer_text))
            span.set_attribute("answer.status", "success")

            return answer_text, model_name

        except Exception as e:
            span.set_attribute("answer.status", "error")
            span.set_attribute("answer.failure_reason", str(e))

            return "Failed to generate answer.", model_name



def answer_query(query: str, retrieval_result: FinalResult, level: int = 1) -> GenerationResult:
    with tracer.start_as_current_span("answer_query") as span:
        if not retrieval_result or not retrieval_result.items:
            span.set_attribute("answer_query.status", "no_retrieval_results")
            return GenerationResult(
                answer="",
                model_used="",
                score=0.0,
                status=GenerationStatus.EMPTY,
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
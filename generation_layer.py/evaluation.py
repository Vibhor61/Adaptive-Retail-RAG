import json
from langchain_groq import ChatGroq
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

groq_llm = ChatGroq(model="llama3-8b-8192")

def evaluate_answer(question: str, answer: str, context: str) -> dict:
    with tracer.start_as_current_span("answer_evaluation") as span:
        try:
            prompt = f"""
                You are an evaluator for an e-commerce product review assistant.
                Given a question, an answer, and the context used to generate the answer, evaluate the quality of the answer.
                
                QUESTION: {question}
                CONTEXT:{context}
                ANSWER:{answer}

                Evaluate the answer and respond in JSON only with this exact format:
                {{
                    "score": <float between 0.0 and 1.0>,
                    "failure_type": "<hallucination|incomplete|irrelevant|refusal|none>",
                    "is_refusal": <true|false>,
                    "root_cause": "<missing_entity_coverage|keyword_miss|semantic_mismatch|none>"
                }}

                Scoring guide:
                - 1.0: answer is fully grounded in context and directly answers the question
                - 0.7-0.9: answer is mostly correct but missing some details
                - 0.4-0.6: answer is partially correct or vague
                - 0.0-0.3: answer is wrong, hallucinated, or completely irrelevant

                Failure types:
                - hallucination: answer contains information not present in context
                - incomplete: answer is too vague or missing key details
                - irrelevant: answer does not address the question
                - refusal: answer says it does not have enough information
                - none: answer is good

                Root cause definitions:
                - missing_entity_coverage: relevant entities/products are missing from context
                - keyword_miss: sparse retrieval failed due to missing keywords
                - semantic_mismatch: dense retrieval returned semantically irrelevant results
                - none: no clear root cause
                Return JSON only, no explanation, no markdown backticks."""

            raw = groq_llm.invoke(prompt)
            
            try:
                result = json.loads(raw.content.strip())
                evaluation_result = {
                    "score": float(result.get("score", 0.0)),
                    "failure_type": result.get("failure_type", "none"),
                    "is_refusal": bool(result.get("is_refusal", False)),
                    "root_cause": result.get("root_cause", "none")
                }
                
                span.set_attribute("evaluation.score", evaluation_result["score"])
                span.set_attribute("evaluation.failure_type", evaluation_result["failure_type"])
                span.set_attribute("evaluation.is_refusal", evaluation_result["is_refusal"])
                span.set_attribute("evaluation.root_cause", evaluation_result["root_cause"])
                span.set_attribute("evaluation.status", "success")
                
                return evaluation_result
            except json.JSONDecodeError:
                span.record_exception(json.JSONDecodeError("Failed to parse evaluation response"))
                span.set_attribute("evaluation.status", "json_error")
                return {
                    "score": 0.0,
                    "failure_type": "incomplete",
                    "is_refusal": False,
                    "root_cause": "none"
                }
        
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("evaluation.status", "error")
            return {
                "score": 0.0,
                "failure_type": "incomplete",
                "is_refusal": False,
                "root_cause": "none"
            }
from langchain_ollama import OllamaLLM
from typing import List
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

llm_ollama = OllamaLLM(model = "mistral:latest")

def rewrite_query(query: str, chat_history: List[str]|None) ->str:
    if not chat_history:
        return query
    with tracer.start_as_current_span("query_rewrite") as span:
        try:
            prompt = f"""
                You are a query rewriter for an e-commerce product review assistant.
                Given the chat history and follow up question, rewrite the follow up as a fully self contained question.

                Chat History: {chat_history}
                Follow up Question: {query}
                
                Return only the rewritten question and nothing else.
            """
            rewritten_query = llm_ollama.invoke(prompt).strip()

            if not rewritten_query or len(rewritten_query) < 3:
                span.set_attribute("requery.status","skipped")

            span.set_attribute("requery.status","success")
            span.set_attribute("requery.length",len(rewritten_query))

            return rewritten_query
        
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("rewrite.status", "error")
            return query
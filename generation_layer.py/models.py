from langchain_ollama import OllamaLLM 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_groq import ChatGroq 

groq_llm = ChatGroq(model="llama3-8b-8192") 
gemini_flash_llm = ChatGoogleGenerativeAI( model="gemini-2.0-flash" ) 
gemini_pro_llm = ChatGoogleGenerativeAI( model="gemini-1.5-pro" ) 

MODEL_REGISTRY = { 
    "small": ( groq_llm, "llama3-8b-8192" ), 
    "medium": ( gemini_flash_llm, "gemini-2.0-flash" ), 
    "large": ( gemini_pro_llm, "gemini-1.5-pro" ) 
} 

def get_model(model_tier: str): 
    return MODEL_REGISTRY[model_tier]
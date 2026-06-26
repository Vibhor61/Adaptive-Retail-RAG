"""
Configuration module for the RAG pipeline settings.
Defines connection details for PostgreSQL, Qdrant, and Phoenix.
Manages application-wide environment variables and derived URLs.
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    qdrant_host: str
    qdrant_port: int


    phoenix_host: str
    phoenix_port: int

    fastapi_host: str
    fastapi_port: int

    groq_api_key: str
    groq_api_keys: Optional[str] = None
    
    generation_model: str
    router_model: str
    rewrite_model: str
    
    embedding_model: str

    reranker_model: str
    
    backend_url: str = "http://localhost:8000"
    phoenix_url_ui: str = "http://localhost:6006"
    ui_file_link: str = "file:///c:/Users/risha/OneDrive/Desktop/RAG/ui.py"
    main_file_link: str = "file:///c:/Users/risha/OneDrive/Desktop/RAG/main.py"

    @property
    def stream_url(self) -> str:
        return f"{self.backend_url}/chat/stream"
        
    class Config:
        env_file = ".env"
        
    @property
    def postgres_url(self):
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_db}"
        )

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


    @property
    def phoenix_url(self) -> str:
        return f"http://{self.phoenix_host}:{self.phoenix_port}"

    @property
    def phoenix_traces_endpoint(self) -> str:
        return f"{self.phoenix_url}/v1/traces"
    
    @property
    def is_groq_generation(self) -> bool:
        return self.generation_model.startswith("groq/")

    @property
    def groq_model_name(self) -> str:
        return self.generation_model.replace("groq/", "")
    
settings = Settings()

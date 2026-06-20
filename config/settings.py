from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    qdrant_host: str
    qdrant_port: int

    ollama_host: str
    ollama_port: int

    phoenix_host: str
    phoenix_port: int

    fastapi_host: str
    fastapi_port: int

    groq_api_key: str
    
    generation_model: str
    router_model: str
    rewrite_model: str
    
    embedding_model: str

    reranker_model: str
    
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
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

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

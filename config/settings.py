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
    
    class Config:
        env_file = ".env"


settings = Settings()

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    # === LLM ===
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # === Tools ===
    tavily_api_key: Optional[str] = Field(None, env="TAVILY_API_KEY")
    
    # === LangSmith / Tracing ===
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field("stock-agent-v2.5", env="LANGCHAIN_PROJECT")
    
    # === FastAPI / Production ===
    secret_key: str = Field("change-this-in-production", env="SECRET_KEY")
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"], env="CORS_ORIGINS")
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # === Self-Improvement ===
    performance_log_path: str = Field("logs/performance.jsonl", env="PERFORMANCE_LOG_PATH")
    min_accuracy_threshold: float = Field(0.52, env="MIN_ACCURACY_THRESHOLD")
    
    # === Email for Code Writing Agent failures ===
    error_email: str = Field("hullc2003@gmail.com", env="ERROR_EMAIL")
    smtp_server: Optional[str] = Field(None, env="SMTP_SERVER")
    smtp_port: int = Field(587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

# Validate critical keys on import
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
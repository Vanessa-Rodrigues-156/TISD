from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings.
    Reads from environment variables or .env file.
    """
    # Model Configuration
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # Conversation Context
    MAX_CONTEXT_LENGTH: int = 10
    CONTEXT_TIMEOUT: int = 3600  # 1 hour
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_PERIOD: str = "minute"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a singleton settings instance
settings = Settings()
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="llm-gateway")
    app_version: str = Field(default="0.1.0")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Data stores
    redis_url: str = Field(default="redis://redis:6379")
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@postgres:5432/llmgateway"
    )

    # Observability
    otel_exporter_otlp_endpoint: str = Field(default="http://jaeger:4318")
    otel_service_name: str = Field(default="llm-gateway")
    log_level: str = Field(default="INFO")

    # LLM provider API keys — stored as SecretStr to avoid accidental logging
    openai_api_key: SecretStr | None = Field(default=None)
    anthropic_api_key: SecretStr | None = Field(default=None)
    together_api_key: SecretStr | None = Field(default=None)
    groq_api_key: SecretStr | None = Field(default=None)

    # LLM call behaviour
    llm_timeout: int = Field(default=60)
    llm_max_retries: int = Field(default=3)


settings = Settings()

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

    # Cache
    cache_ttl: int = Field(
        default=3600, description="TTL in seconds for cached responses (default 1 hour)"
    )

    # Semantic cache
    enable_semantic_cache: bool = Field(
        default=True, description="Enable embedding-based semantic similarity cache"
    )
    semantic_cache_threshold: float = Field(
        default=0.95,
        description="Minimum cosine similarity [0, 1] to count as a semantic hit",
    )
    semantic_cache_max_entries: int = Field(
        default=1000,
        description="Maximum number of embeddings to store per model in the semantic index",
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable token bucket rate limiting")
    rate_limit_default_rate: float = Field(
        default=10.0,
        description="Default token refill rate in requests per minute",
    )
    rate_limit_default_capacity: int = Field(
        default=20,
        description="Default token bucket capacity (maximum burst size)",
    )

    # Cost tracking
    daily_cost_alert_threshold: float = Field(
        default=10.0,
        description="Log a warning when daily spend (USD) exceeds this value",
    )

    # Gateway API key — protects /v1/chat/completions when set
    gateway_api_key: str | None = Field(
        default=None,
        description="Secret key required in Authorization: Bearer header for completions endpoint",
    )

    # Admin API
    admin_api_key: str | None = Field(
        default=None,
        description="Secret key required in X-Admin-Key header for admin endpoints",
    )


settings = Settings()

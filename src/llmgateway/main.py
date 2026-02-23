import os

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import make_asgi_app

from llmgateway.api.admin import router as admin_router
from llmgateway.api.completions import router as completions_router
from llmgateway.api.health import router as health_router
from llmgateway.cache import CacheManager, EmbeddingModel, RedisCache
from llmgateway.config import settings
from llmgateway.cost import CostTracker
from llmgateway.providers import LLMGatewayProvider
from llmgateway.ratelimit import RateLimiter

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.stdlib.NAME_TO_LEVEL.get(settings.log_level.upper(), 20)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------
resource = Resource.create({"service.name": settings.otel_service_name})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(
    endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces",
)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Gateway",
    version=settings.app_version,
    description=(
        "Production-grade LLM Gateway providing multi-provider routing, "
        "semantic caching, rate limiting, and full observability."
    ),
)

# CORS — restrictive defaults; override via environment in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint mounted as a sub-application
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routers
app.include_router(health_router)
app.include_router(completions_router)
app.include_router(admin_router)

# Instrument *after* routes are registered
FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    # Propagate API keys from Settings into os.environ so LiteLLM can read them.
    # pydantic-settings reads the .env file into the Settings object but does
    # NOT automatically populate os.environ.
    _key_map = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "TOGETHER_API_KEY": settings.together_api_key,
        "GROQ_API_KEY": settings.groq_api_key,
    }
    for env_var, secret in _key_map.items():
        if secret is not None:
            os.environ[env_var] = secret.get_secret_value()

    # Initialise the shared provider and attach it to app state so the
    # get_provider() dependency can inject it into every request handler.
    app.state.provider = LLMGatewayProvider(
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )

    # Initialise the cost tracker.  A connection error is non-fatal; the
    # gateway will serve requests without persisting usage records.
    try:
        app.state.cost_tracker = CostTracker(database_url=settings.database_url)
        log.info("cost_tracker_initialized", database_url=settings.database_url)
    except Exception as exc:
        log.warning("cost_tracker_init_failed", error=str(exc))
        app.state.cost_tracker = None

    # Initialise Redis-backed CacheManager.  A connection error here is
    # non-fatal: the gateway will still serve requests, just without caching.
    try:
        redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        embedding_model = EmbeddingModel() if settings.enable_semantic_cache else None
        app.state.cache_manager = CacheManager(
            backend=RedisCache(redis_client),
            default_ttl=settings.cache_ttl,
            redis_client=redis_client,
            embedding_model=embedding_model,
            semantic_threshold=settings.semantic_cache_threshold,
            semantic_max_entries=settings.semantic_cache_max_entries,
        )
        app.state.redis_client = redis_client

        # Rate limiter shares the same Redis connection.
        app.state.rate_limiter = RateLimiter(
            redis_client=redis_client,
            default_capacity=float(settings.rate_limit_default_capacity),
            default_rate=settings.rate_limit_default_rate / 60.0,
            enabled=settings.rate_limit_enabled,
        )

        log.info(
            "cache_initialized",
            redis_url=settings.redis_url,
            cache_ttl=settings.cache_ttl,
            semantic_cache=settings.enable_semantic_cache,
            semantic_threshold=settings.semantic_cache_threshold,
            rate_limiting=settings.rate_limit_enabled,
            rate_limit_capacity=settings.rate_limit_default_capacity,
            rate_limit_rate=settings.rate_limit_default_rate,
        )
    except Exception as exc:
        log.warning("cache_init_failed", error=str(exc))
        app.state.cache_manager = None
        app.state.redis_client = None
        app.state.rate_limiter = None

    log.info(
        "LLM Gateway ready",
        host=settings.host,
        port=settings.port,
        llm_timeout=settings.llm_timeout,
        llm_max_retries=settings.llm_max_retries,
        redis_url=settings.redis_url,
        otel_endpoint=settings.otel_exporter_otlp_endpoint,
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    log.info("LLM Gateway shutting down")
    redis_client = getattr(app.state, "redis_client", None)
    if redis_client is not None:
        await redis_client.aclose()
    cost_tracker = getattr(app.state, "cost_tracker", None)
    if cost_tracker is not None:
        await cost_tracker.close()
    tracer_provider.shutdown()

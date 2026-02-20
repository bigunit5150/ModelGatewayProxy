import os

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

from llmgateway.api.completions import router as completions_router
from llmgateway.api.health import router as health_router
from llmgateway.config import settings
from llmgateway.providers import LLMGatewayProvider

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
    tracer_provider.shutdown()

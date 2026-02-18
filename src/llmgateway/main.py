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

from llmgateway.api.health import router as health_router
from llmgateway.config import settings

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

# Instrument *after* routes are registered
FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    log.info(
        "LLM Gateway started",
        host=settings.host,
        port=settings.port,
        redis_url=settings.redis_url,
        otel_endpoint=settings.otel_exporter_otlp_endpoint,
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    log.info("LLM Gateway shutting down")
    tracer_provider.shutdown()

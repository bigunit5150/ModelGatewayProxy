# Production Deployment — Fly.io

This guide walks through deploying the LLM Gateway to
[Fly.io](https://fly.io) with managed Redis and PostgreSQL.

---

## Prerequisites

| Tool | Install |
|---|---|
| `flyctl` CLI | `curl -L https://fly.io/install.sh \| sh` |
| Docker (for local builds) | https://docs.docker.com/get-docker/ |
| GitHub repository | Push the code before running the deploy pipeline |

---

## One-time Setup

### 1. Authenticate

```bash
flyctl auth login
```

### 2. Create the app

Run this once from the repo root. Answer **No** when asked to deploy
immediately — we need to attach databases first.

```bash
flyctl launch --no-deploy --name llm-gateway-ksmith --region iad
```

> `fly.toml` is already committed; `flyctl launch` will detect it and skip
> regenerating it.

### 3. Create a persistent volume for the embedding model cache

The sentence-transformers model (~90 MB) is downloaded on first startup.
A volume prevents re-downloading it on every deploy.

```bash
flyctl volumes create model_cache --size 2 --region iad
```

### 4. Create managed Redis (Upstash)

```bash
flyctl redis create --name llm-gateway-redis --region iad
```

Copy the `redis://` URL from the output, then set it as a secret:

```bash
flyctl secrets set REDIS_URL="redis://default:<password>@<host>:<port>"
```

### 5. Create managed PostgreSQL

```bash
flyctl postgres create --name llm-gateway-db --region iad
```

Attach the cluster to the app. This automatically sets the `DATABASE_URL`
secret in the correct `postgresql+asyncpg://` format:

```bash
flyctl postgres attach llm-gateway-db
```

> **Note:** `flyctl postgres attach` sets `DATABASE_URL` using the
> `postgresql://` scheme. The gateway requires `postgresql+asyncpg://`.
> Verify with `flyctl secrets list` and update if needed:
> ```bash
> flyctl secrets set DATABASE_URL="postgresql+asyncpg://postgres:<pass>@<host>/<db>"
> ```

### 6. Set LLM provider API keys and admin secret

```bash
flyctl secrets set \
  OPENAI_API_KEY="sk-..." \
  ANTHROPIC_API_KEY="sk-ant-..." \
  TOGETHER_API_KEY="..." \
  GROQ_API_KEY="..." \
  ADMIN_API_KEY="$(openssl rand -hex 32)"
```

Only set the keys for providers you intend to use.

### 7. Deploy

```bash
flyctl deploy
```

Fly.io will:
1. Build the Docker image remotely (`--remote-only` is set in CI; local
   builds work too).
2. Run `alembic upgrade head` in a temporary release VM to apply any
   pending database migrations.
3. Roll out the new version across both machines with zero downtime.

---

## GitHub Actions CI/CD

The workflow in `.github/workflows/ci.yml` runs automatically:

| Trigger | Jobs |
|---|---|
| Pull request to `main`/`master` | `test` (lint, type-check, unit tests) |
| Push to `main`/`master` | `test` → `deploy` |

### Required GitHub secret

Add one secret to the repository under **Settings → Secrets and variables →
Actions**:

| Secret | Value |
|---|---|
| `FLY_API_TOKEN` | Output of `flyctl auth token` |

---

## Scaling

### Vertical (more CPU / RAM per machine)

Edit `[[vm]]` in `fly.toml` and redeploy:

```toml
[[vm]]
  cpus = 4
  memory_mb = 2048
```

Also increase `--workers` in the Dockerfile `CMD` to match:

```dockerfile
CMD ["uvicorn", "llmgateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Horizontal (more machines)

```bash
flyctl scale count 3
```

The token-bucket rate limiter and semantic cache both use Redis as the
shared backend, so horizontal scaling is safe.

---

## Observability

### Logs

```bash
flyctl logs                  # tail live logs
flyctl logs --instance <id>  # single machine
```

### Prometheus metrics

Fly.io scrapes `/metrics` automatically (configured in `fly.toml`). You can
also scrape it directly:

```bash
curl https://llm-gateway-ksmith.fly.dev/metrics
```

Key metrics to watch:

| Metric | Description |
|---|---|
| `llm_requests_total` | Request rate by model / provider / status |
| `llm_request_duration_seconds` | End-to-end latency (p50/p95/p99) |
| `llm_cost_usd_total` | Running cost per model / user |
| `llm_rate_limit_exceeded_total` | Rate-limit violations per user |
| `llm_cache_hits_total` | Exact cache hit rate |
| `llm_semantic_cache_hits_total` | Semantic cache hit rate |

### Health endpoints

```bash
curl https://llm-gateway-ksmith.fly.dev/health          # basic liveness
curl https://llm-gateway-ksmith.fly.dev/health/ready    # Redis + Postgres check
```

### Distributed traces (OpenTelemetry)

Point the collector at a reachable endpoint and set the secret:

```bash
flyctl secrets set OTEL_EXPORTER_OTLP_ENDPOINT="https://your-collector:4318"
```

---

## Database Migrations

Migrations run automatically as a release command before each deploy.
To run them manually (e.g. to roll back):

```bash
# Open a shell in a running machine
flyctl ssh console

# Inside the machine
alembic upgrade head        # apply all pending
alembic downgrade -1        # roll back one revision
alembic current             # show current revision
```

---

## Secrets Reference

| Secret | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | `postgresql+asyncpg://...` — set by `postgres attach` |
| `REDIS_URL` | Yes | `redis://...` — set after `redis create` |
| `OPENAI_API_KEY` | If using OpenAI | `sk-...` |
| `ANTHROPIC_API_KEY` | If using Anthropic | `sk-ant-...` |
| `TOGETHER_API_KEY` | If using Together AI | — |
| `GROQ_API_KEY` | If using Groq | — |
| `ADMIN_API_KEY` | Recommended | Protects `/admin/costs/summary` |
| `FLY_API_TOKEN` | CI/CD only | `flyctl auth token` → GitHub secret |

---

## Troubleshooting

**Startup fails with model download error**
The sentence-transformers model requires network access on first boot. If the
volume (`model_cache`) is not attached, the model re-downloads every deploy.
Verify the volume is mounted:
```bash
flyctl volumes list
```

**`alembic upgrade head` fails in release command**
Check the release logs:
```bash
flyctl releases --image  # find the failed release
flyctl logs --instance <release-vm-id>
```
The most common cause is an incorrect `DATABASE_URL` scheme (`postgresql://`
instead of `postgresql+asyncpg://`).

**Machines not reaching `/health/ready`**
The readiness probe connects to Redis and Postgres. If either is unreachable
(wrong URL, firewall rule, or service not ready), the machine stays out of
the load balancer rotation. Check `flyctl logs` and verify the secrets:
```bash
flyctl secrets list
```

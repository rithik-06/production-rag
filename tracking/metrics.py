from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from loguru import logger

# ── Define Metrics ─────────────────────────────────────

# Count total queries received
QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total number of queries received"
)

# Count cache hits vs misses
CACHE_HIT_COUNTER = Counter(
    "rag_cache_hits_total",
    "Total number of cache hits"
)
CACHE_MISS_COUNTER = Counter(
    "rag_cache_misses_total",
    "Total number of cache misses"
)

# Track response time
RESPONSE_TIME = Histogram(
    "rag_response_time_seconds",
    "Response time in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# Track faithfulness score
FAITHFULNESS_GAUGE = Gauge(
    "rag_faithfulness_score",
    "Latest RAGAS faithfulness score"
)

# Track total documents ingested
DOCS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total number of documents ingested"
)


def get_metrics_app():
    """Returns a Prometheus metrics ASGI app to mount on FastAPI."""
    return make_asgi_app()
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger
import shutil
import os
import time
from dotenv import load_dotenv

from prometheus_client import make_asgi_app
from tracking.metrics import (
    QUERY_COUNTER,
    CACHE_HIT_COUNTER,
    CACHE_MISS_COUNTER,
    RESPONSE_TIME,
    FAITHFULNESS_GAUGE,
    DOCS_INGESTED
)

from ingestion.loader import load_all
from chunking.chunker import chunk_documents
from embeddings.embedder import load_embedder, embed_chunks
from retrieval.vector_store import get_client, create_collection, store_chunks, search
from generation.generator import generate_answer
from cache.cache import get_client as get_redis_client, get_cached_answer, set_cached_answer
from tracking.tracker import init_mlflow, log_run

load_dotenv()

# ── App Setup ─────────────────────────────────────────
app = FastAPI(
    title="Production RAG API",
    description="Production-grade RAG pipeline with evaluation and monitoring",
    version="1.0.0"
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Initialize all clients on startup ─────────────────
embedder = None
qdrant_client = None
redis_client = None

@app.on_event("startup")
async def startup():
    global embedder, qdrant_client, redis_client
    logger.info("Starting up RAG API...")

    embedder = load_embedder()

    qdrant_client = get_client()
    create_collection(qdrant_client, "documents", vector_size=384)

    redis_client = get_redis_client()

    init_mlflow()

    logger.info("RAG API ready ✅")


# ── Request/Response Models ────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    question: str
    answer: str
    cached: bool


# ── Endpoints ─────────────────────────────────────────
@app.get("/health")
def health():
    """Check if API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF document and ingest it into the RAG pipeline."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"File uploaded: {file.filename}")

    docs = load_all(folder_path="data")
    chunks = chunk_documents(docs)
    vectors = embed_chunks(chunks, embedder)
    store_chunks(qdrant_client, "documents", chunks, vectors)

    DOCS_INGESTED.inc(len(chunks))

    return {
        "message": f"Successfully ingested {file.filename}",
        "chunks_created": len(chunks)
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question and get an answer from the RAG pipeline."""
    question = request.question
    logger.info(f"Query received: {question}")

    QUERY_COUNTER.inc()
    start = time.time()

    # Check cache first
    cached_answer = get_cached_answer(redis_client, question)
    if cached_answer:
        CACHE_HIT_COUNTER.inc()
        return QueryResponse(
            question=question,
            answer=cached_answer,
            cached=True
        )

    CACHE_MISS_COUNTER.inc()

    # Run full pipeline
    query_vector = embedder.embed_query(question)
    results = search(qdrant_client, "documents", query_vector, top_k=request.top_k)
    answer = generate_answer(question, results)

    # Cache the answer
    set_cached_answer(redis_client, question, answer)

    # Track response time
    RESPONSE_TIME.observe(time.time() - start)

    return QueryResponse(
        question=question,
        answer=answer,
        cached=False
    )
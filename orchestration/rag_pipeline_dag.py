from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# ── Default DAG settings ───────────────────────────────
default_args = {
    "owner": "rithik",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

PROJECT_DIR = "/home/rithik/AI projects/production-rag"
VENV_PYTHON = f"{PROJECT_DIR}/p-rag/bin/python"

# ── DAG Definition ─────────────────────────────────────
with DAG(
    dag_id="rag_pipeline",
    default_args=default_args,
    description="Production RAG pipeline — ingest, embed, evaluate",
    schedule_interval="0 2 * * *",  # runs every night at 2am
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["rag", "production"]
) as dag:

    # Task 1 — Ingest + Chunk
    task_ingest = BashOperator(
        task_id="ingest_and_chunk",
        bash_command=f"cd '{PROJECT_DIR}' && {VENV_PYTHON} -c \
            'from ingestion.loader import load_all; \
             from chunking.chunker import chunk_documents; \
             docs = load_all(folder_path=\"data\"); \
             chunks = chunk_documents(docs); \
             print(f\"✅ Chunks created: {{len(chunks)}}\")'",
    )

    # Task 2 — Embed + Store
    task_embed = BashOperator(
        task_id="embed_and_store",
        bash_command=f"cd '{PROJECT_DIR}' && {VENV_PYTHON} -c \
            'from ingestion.loader import load_all; \
             from chunking.chunker import chunk_documents; \
             from embeddings.embedder import load_embedder, embed_chunks; \
             from retrieval.vector_store import get_client, create_collection, store_chunks; \
             docs = load_all(folder_path=\"data\"); \
             chunks = chunk_documents(docs); \
             embedder = load_embedder(); \
             vectors = embed_chunks(chunks, embedder); \
             client = get_client(); \
             client.delete_collection(\"documents\"); \
             create_collection(client, \"documents\", vector_size=384); \
             store_chunks(client, \"documents\", chunks, vectors); \
             print(\"✅ Stored in Qdrant\")'",
    )

    # Task 3 — Evaluate
    task_evaluate = BashOperator(
        task_id="evaluate_pipeline",
        bash_command=f"cd '{PROJECT_DIR}' && {VENV_PYTHON} test_ingestion.py",
    )

    # Task order
    task_ingest >> task_embed >> task_evaluate
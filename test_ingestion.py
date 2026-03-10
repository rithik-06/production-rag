from ingestion.loader import load_all
from chunking.chunker import chunk_documents
from embeddings.embedder import load_embedder, embed_chunks
from retrieval.vector_store import get_client, create_collection, store_chunks, search
from generation.generator import generate_answer
from cache.cache import get_client as get_redis_client, get_cached_answer, set_cached_answer
from evaluation.evaluator import evaluate_rag
from tracking.tracker import init_mlflow, log_run
from loguru import logger

# Initialize MLflow
init_mlflow()

# Load → Chunk → Embed → Store
docs = load_all(folder_path="data")
chunks = chunk_documents(docs)
embedder = load_embedder()
vectors = embed_chunks(chunks, embedder)
client = get_client()
client.delete_collection("resume")
create_collection(client, "resume", vector_size=384)
store_chunks(client, "resume", chunks, vectors)

# Redis client
redis_client = get_redis_client()

# Test questions
questions = [
    "What is Rithik's education?",
    "What are Rithik's skills?",
    "What projects has Rithik built?"
]

answers = []
contexts = []

# Run pipeline for each question
for question in questions:
    query_vector = embedder.embed_query(question)
    results = search(client, "resume", query_vector, top_k=3)
    answer = generate_answer(question, results)
    answers.append(answer)
    contexts.append([r.payload["text"] for r in results])
    print(f"\n Q: {question}")
    print(f" A: {answer}")

# Evaluate
print("\n⏳ Running RAGAS evaluation...")
scores = evaluate_rag(questions, answers, contexts)
print(f"\n📊 RAGAS Scores:")
print(f"Faithfulness: {scores['faithfulness']}")

# Log to MLflow
log_run(
    params={
        "chunk_size": 256,
        "chunk_overlap": 64,
        "embedding_model": "intfloat/e5-small-v2",
        "llm_model": "llama-3.1-8b-instant",
        "top_k": 3
    },
    metrics=scores,
    run_name="baseline-run"
)

print("\n✅ Run logged to MLflow!")
print("👉 Check it at: http://localhost:5000")

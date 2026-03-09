from ingestion.loader import load_all
from chunking.chunker import chunk_documents
from embeddings.embedder import load_embedder, embed_chunks
from retrieval.vector_store import get_client, create_collection, store_chunks, search
from generation.generator import generate_answer
from loguru import logger
from cache.cache import get_client as get_redis_client, get_cached_answer, set_cached_answer
import time


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

question = "What is Rithik's education?"

# First request — should be a cache MISS
print("\n--- First Request ---")
start = time.time()
cached = get_cached_answer(redis_client, question)
if cached:
    answer = cached
else:
    query_vector = embedder.embed_query(question)
    results = search(client, "resume", query_vector, top_k=3)
    answer = generate_answer(question, results)
    set_cached_answer(redis_client, question, answer)
print(f"Answer: {answer}")
print(f"Time: {time.time() - start:.2f}s")

# Second request — should be a cache HIT (much faster!)
print("\n--- Second Request (same question) ---")
start = time.time()
cached = get_cached_answer(redis_client, question)
if cached:
    answer = cached
print(f"Answer: {answer}")
print(f"Time: {time.time() - start:.2f}s")
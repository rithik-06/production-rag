from ingestion.loader import load_all
from chunking.chunker import chunk_documents
from embeddings.embedder import load_embedder, embed_chunks
from retrieval.vector_store import get_client, create_collection, store_chunks, search
from generation.generator import generate_answer
from loguru import logger


# Load
docs = load_all(folder_path="data")

# Chunk
chunks = chunk_documents(docs)

# Embed
embedder = load_embedder()
vectors = embed_chunks(chunks, embedder)

# Store in Qdrant - delete first to avoid duplicates
client = get_client()

# Delete old collection if exists
client.delete_collection("resume")
logger.info("Old collection deleted ✅")

create_collection(client, "resume", vector_size=384)
store_chunks(client, "resume", chunks, vectors)

# Search
question = "What is Rithik's education?"
query_vector = embedder.embed_query(question)
results = search(client, "resume", query_vector, top_k=5)

# DEBUG - see what chunks are retrieved
print("\n📄 Retrieved chunks:")
for i, r in enumerate(results):
    print(f"\n--- Chunk {i+1} (score: {r.score:.3f}) ---")
    print(r.payload["text"][:200])

# Generate answer
answer = generate_answer(question, results)
print(f"\n🔍 Question: {question}")
print(f"\n🤖 Answer:\n{answer}")
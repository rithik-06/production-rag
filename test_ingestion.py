from ingestion.loader import load_all
from chunking.chunker import chunk_documents
from embeddings.embedder import load_embedder, embed_chunks
from retrieval.vector_store import get_client, create_collection, store_chunks, search
from generation.generator import generate_answer

# Load
docs = load_all(folder_path="data")
print(f"Pages loaded: {len(docs)}")

# Chunk
chunks = chunk_documents(docs)
print(f"Total chunks: {len(chunks)}")
print(f"\nChunk 1:\n{chunks[0].page_content}")
print(f"\nChunk 2:\n{chunks[1].page_content}")

# Embed  ← is this part there?
embedder = load_embedder()
vectors = embed_chunks(chunks, embedder)

print(f"Total vectors: {len(vectors)}")
print(f"Vector size: {len(vectors[0])} dimensions")
print(f"First 5 numbers of vector 1: {vectors[0][:5]}")

# Store in Qdrant
client = get_client()
create_collection(client, "resume", vector_size=384)
store_chunks(client, "resume", chunks, vectors)

# Search — ask a question
query = "What are Rithik's skills?"
query_vector = embedder.embed_query(query)
results = search(client, "resume", query_vector, top_k=3)

print(f"\n🔍 Query: {query}")
print(f"\nTop 3 relevant chunks:")
for i, r in enumerate(results):
    print(f"\n--- Result {i+1} (score: {r.score:.3f}) ---")
    print(r.payload["text"])

# Generate answer
question = "What are Rithik's skills?"
question = "What is Rithik's education?"
answer = generate_answer(question, results)

print(f"\n🔍 Question: {question}")
print(f"\n🤖 Answer:\n{answer}")
question = "What are Rithik's skills?"
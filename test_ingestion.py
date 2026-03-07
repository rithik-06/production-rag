from ingestion.loader import load_all
from chunking.chunker import chunk_documents
from embeddings.embedder import load_embedder, embed_chunks

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
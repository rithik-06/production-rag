from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger
import uuid


def get_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """Connect to Qdrant server."""
    client = QdrantClient(host=host, port=port)
    logger.info(f"Connected to Qdrant at {host}:{port} ✅")
    return client


def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 384):
    """
    Create a collection in Qdrant.
    Collection = like a table in a normal database
    but specifically designed to store vectors.
    """
    # Check if collection already exists
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists ✅")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,      # must match our embedding size (384)
            distance=Distance.COSINE  # compare vectors using cosine similarity
        )
    )
    logger.info(f"Collection '{collection_name}' created ✅")


def store_chunks(client: QdrantClient, collection_name: str, chunks: list, vectors: list):
    """
    Store chunks + their vectors into Qdrant.
    Each point = 1 chunk + its vector + original text as metadata.
    """
    logger.info(f"Storing {len(chunks)} chunks into Qdrant...")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),         # unique ID for each chunk
            vector=vectors[i],             # the embedding vector
            payload={"text": chunks[i].page_content,  # original text
                     "source": chunks[i].metadata.get("source", "unknown")}
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Stored {len(points)} points in Qdrant ✅")


def search(client: QdrantClient, collection_name: str, query_vector: list, top_k: int = 3):
    """
    Search for most similar chunks to a query vector.
    Returns top_k most relevant chunks.
    """
    logger.info(f"Searching for top {top_k} similar chunks...")

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points

    logger.info(f"Found {len(results)} results ✅")
    return results
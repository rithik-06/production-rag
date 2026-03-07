from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger


def load_embedder(model_name: str = "intfloat/e5-small-v2"):
    """
    Loads the embedding model.
    e5-small-v2 is lightweight but powerful — perfect for development.
    We can swap to e5-large-v2 in production for better accuracy.
    """
    logger.info(f"Loading embedding model: {model_name}")

    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  
        # normalize = keeps all vectors same scale
        # makes similarity comparison more accurate
    )

    logger.info("Embedding model loaded ✅")
    return embedder


def embed_chunks(chunks: list, embedder) -> list:
    """
    Takes our chunks and converts each one into a vector (list of numbers).
    Returns the chunks with embeddings attached.
    """
    logger.info(f"Embedding {len(chunks)} chunks...")

    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_documents(texts)

    logger.info(f"Generated {len(vectors)} vectors")
    logger.info(f"Vector size: {len(vectors[0])} dimensions")

    return vectors
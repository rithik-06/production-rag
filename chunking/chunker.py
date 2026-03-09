from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def chunk_documents(documents: list, chunk_size: int = 512, chunk_overlap: int = 64) -> list:
    """
    Splits documents into smaller chunks.
    
    chunk_size    = max characters per chunk
    chunk_overlap = how many characters overlap between chunks
                    (so we don't lose context at boundaries)
    """
    logger.info(f"Chunking {len(documents)} documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # It tries to split in this order:
        # 1. paragraph breaks
        # 2. new lines  
        # 3. sentences
        # 4. words
        # 5. characters (last resort)
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"Total chunks created: {len(chunks)}")
    logger.info(f"Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks
# langchain has ready-made loaders so we don't build from scratch
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from pathlib import Path
from loguru import logger


def load_pdfs(folder_path: str) -> list:
    """
    Goes into a folder, finds all PDFs, extracts text from each page.
    Returns a list of documents.
    """
    documents = []
    folder = Path(folder_path)

    # Loop through every PDF file in the folder
    for pdf_file in folder.glob("*.pdf"):
        logger.info(f"Loading PDF: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    logger.info(f"Total pages loaded from PDFs: {len(documents)}")
    return documents


def load_urls(urls: list) -> list:
    """
    Takes a list of URLs, scrapes each one, extracts readable text.
    Returns a list of documents.
    """
    logger.info(f"Loading {len(urls)} URLs")
    loader = WebBaseLoader(urls)
    documents = loader.load()
    logger.info(f"Total pages loaded from web: {len(documents)}")
    return documents


def load_all(folder_path: str = None, urls: list = None) -> list:
    """
    Master function — loads from all sources and combines everything.
    This is what other modules will call.
    """
    all_documents = []

    if folder_path:
        all_documents.extend(load_pdfs(folder_path))

    if urls:
        all_documents.extend(load_urls(urls))

    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents
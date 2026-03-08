from groq import Groq
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


def get_client():
    """Connect to Groq."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Connected to Groq ✅")
    return client


def generate_answer(question: str, retrieved_chunks: list) -> str:
    """
    Takes the user question + retrieved chunks from Qdrant
    and generates a proper answer using LLM.

    This is the core of RAG:
    R = Retrieved chunks (from Qdrant)
    A = Augmented prompt (chunks + question)
    G = Generated answer (from LLM)
    """
    client = get_client()

    # Build context from retrieved chunks
    # We join all chunks into one block of text
    context = "\n\n".join([chunk.payload["text"] for chunk in retrieved_chunks])

    # This is called a "prompt template"
    # We tell the LLM exactly how to behave
    prompt = f"""You are a helpful assistant. 
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

    logger.info(f"Generating answer for: {question}")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # fast and free on Groq
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,   # low temperature = focused, factual answers
        max_tokens=512
    )

    answer = response.choices[0].message.content
    logger.info("Answer generated ✅")
    return answer
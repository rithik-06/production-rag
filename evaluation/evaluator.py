from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


def evaluate_rag(questions: list, answers: list, contexts: list) -> dict:
    """
    Evaluates RAG pipeline quality using RAGAS metrics.
    """
    logger.info("Starting RAGAS evaluation...")

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    dataset = Dataset.from_dict(data)

    # Use Groq as the judge LLM
    groq_llm = LangchainLLMWrapper(
    ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
)

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    )

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=groq_llm,
        embeddings=embeddings
    )

    # Convert results to pandas to extract scores safely
    df = results.to_pandas()

    scores = {
        "faithfulness": round(df["faithfulness"].mean(), 3),
        
    }

    logger.info(f"Evaluation complete ✅")
    logger.info(f"Faithfulness:     {scores['faithfulness']}")
    

    return scores
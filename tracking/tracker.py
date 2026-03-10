import mlflow
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def init_mlflow(experiment_name: str = "production-rag"):
    """
    Connect to MLflow and set the experiment.
    Experiment = a group of related runs.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow initialized ✅ experiment: {experiment_name}")


def log_run(params: dict, metrics: dict, run_name: str = "rag-run"):
    """
    Log a single run to MLflow.

    params  = settings we used (chunk_size, model, top_k...)
    metrics = scores we got (faithfulness, answer_relevancy...)
    """
    with mlflow.start_run(run_name=run_name):

        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            if value == value:  # skip NaN values
                mlflow.log_metric(key, value)

        logger.info(f"Run logged to MLflow ✅")
        logger.info(f"Params:  {params}")
        logger.info(f"Metrics: {metrics}")
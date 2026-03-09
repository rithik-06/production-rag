import redis
import json
import hashlib
from loguru import logger


def get_client(host: str = "localhost", port: int = 6379):
    """Connect to Redis."""
    client = redis.Redis(host=host, port=port, decode_responses=True)
    logger.info("Connected to Redis ✅")
    return client


def make_key(question: str) -> str:
    """
    Convert question into a unique cache key.
    We use MD5 hash so long questions become short keys.

    "What are Rithik's skills?" → "a3f8c2d1..."
    """
    return "rag:" + hashlib.md5(question.encode()).hexdigest()


def get_cached_answer(client, question: str) -> str | None:
    """
    Check if we already have an answer for this question.
    Returns answer if found, None if not.
    """
    key = make_key(question)
    cached = client.get(key)

    if cached:
        logger.info(f"Cache HIT ✅ for: {question}")
        return json.loads(cached)

    logger.info(f"Cache MISS ❌ for: {question}")
    return None


def set_cached_answer(client, question: str, answer: str, ttl: int = 3600):
    """
    Save answer to Redis cache.
    ttl = time to live in seconds (3600 = 1 hour)
    After 1 hour, Redis deletes it automatically.
    """
    key = make_key(question)
    client.setex(key, ttl, json.dumps(answer))
    logger.info(f"Answer cached for 1 hour ✅")
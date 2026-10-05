"""Shared configuration for Lab 18."""

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & LLM Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Auto-detect Gemini key if in OPENAI_API_KEY or GEMINI_API_KEY
if not GEMINI_API_KEY and OPENAI_API_KEY.startswith("AIza"):
    GEMINI_API_KEY = OPENAI_API_KEY

if GEMINI_API_KEY:
    LLM_API_KEY = GEMINI_API_KEY
    LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-lite-latest")
    os.environ["OPENAI_API_KEY"] = GEMINI_API_KEY
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
else:
    LLM_API_KEY = OPENAI_API_KEY
    LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    LLM_MODEL = "gpt-4o-mini"


import time
import threading

_last_call_time = 0.0
_rate_lock = threading.Lock()


def rate_limit_sleep(min_interval: float = 4.2):
    global _last_call_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_call_time = time.time()


def get_openai_client():
    from openai import OpenAI
    if LLM_BASE_URL:
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return OpenAI(api_key=LLM_API_KEY)


def get_chat_model(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    rate_limiter = InMemoryRateLimiter(requests_per_second=0.23, check_every_n_seconds=0.1, max_bucket_size=1)
    kwargs = {
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY,
        "temperature": temperature,
        "rate_limiter": rate_limiter,
        "max_retries": 5,
    }
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return ChatOpenAI(**kwargs)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

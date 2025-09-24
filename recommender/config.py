import os

# Logging & debug
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG = os.getenv("DEBUG", "0") == "1"

# Services
OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT = os.getenv("QDRANT_URL", "http://qdrant:6333")
QCOLL = os.getenv("QDRANT_COLLECTION", "products_poc")
LLAMA = os.getenv("LLAMA_STACK_URL", "http://llama-stack:8080")
MODEL = os.getenv("MODEL_ID", "llama3.2:3b")

# Recommender knobs
MAX_RESULTS = int(os.getenv("RECO_MAX_RESULTS", "10"))
SCORE_THRESHOLD = float(os.getenv("RECO_SCORE_THRESHOLD", "0.01"))

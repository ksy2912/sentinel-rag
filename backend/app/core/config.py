import os


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

ALLOW_MOCK_LLM = _bool("ALLOW_MOCK_LLM")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_RETRY_BUDGET = int(os.getenv("RAG_RETRY_BUDGET", "2"))
RAG_GROUNDING_THRESHOLD = float(os.getenv("RAG_GROUNDING_THRESHOLD", "0.5"))
RAG_RETRIEVAL_QUALITY_THRESHOLD = float(os.getenv("RAG_RETRIEVAL_QUALITY_THRESHOLD", "0.25"))

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

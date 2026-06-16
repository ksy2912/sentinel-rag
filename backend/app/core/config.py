import os


def _get_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/rag",
)

# --- LLM settings ---
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_HTTP_REFERER: str | None = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_TITLE: str | None = os.getenv("OPENROUTER_APP_TITLE")

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3")

ALLOW_MOCK_LLM: bool = _get_bool("ALLOW_MOCK_LLM", "false")
LLM_RETRY_LIMIT: int = int(os.getenv("LLM_RETRY_LIMIT", "2"))

# --- Embeddings settings ---
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

# --- RAG / agent settings ---
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))
RAG_RETRY_BUDGET: int = int(os.getenv("RAG_RETRY_BUDGET", "2"))
RAG_GROUNDING_THRESHOLD: float = float(os.getenv("RAG_GROUNDING_THRESHOLD", "0.5"))
RAG_RETRIEVAL_QUALITY_THRESHOLD: float = float(os.getenv("RAG_RETRIEVAL_QUALITY_THRESHOLD", "0.25"))

# --- Observability (Day 11) ---
LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


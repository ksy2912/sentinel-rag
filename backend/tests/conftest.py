import os

os.environ.setdefault("ALLOW_MOCK_LLM", "true")
os.environ.setdefault("DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag")

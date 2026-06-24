from __future__ import annotations

import os
import tempfile

from fastapi import HTTPException
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_LOADERS = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}


def extract_text(raw: bytes, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1] or ".txt"
    if ext not in _LOADERS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        docs = _LOADERS[ext](path).load()
        text = "\n\n".join(d.page_content.strip() for d in docs if d.page_content.strip())
        if not text:
            raise HTTPException(status_code=400, detail="File has no readable text.")
        return text
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc
    finally:
        os.unlink(path)


def split_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    if not text.strip():
        return []
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    ).split_text(text.strip())

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _group_lines(lines: list[str], max_size: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_size:
            groups.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        groups.append("\n".join(current))
    return groups


def _window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def _split_long_block(block: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(block) <= chunk_size:
        return [block]
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if len(lines) > 1:
        return _group_lines(lines, chunk_size)
    return _window_chunks(block, chunk_size, chunk_overlap)


def split_into_overlapping_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Generic chunking for any document:
    - split on paragraph breaks
    - build overlapping chunks by size
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    expanded: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            expanded.append(para)
        else:
            expanded.extend(_split_long_block(para, chunk_size, chunk_overlap))
    paragraphs = expanded
    chunks: list[str] = []

    current: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        chunk_text = "\n\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if chunk_overlap <= 0:
            current = []
            current_len = 0
            return

        overlap_paras: list[str] = []
        overlap_len = 0
        for p in reversed(current):
            p_len = len(p) + 2
            if overlap_len >= chunk_overlap:
                break
            overlap_paras.insert(0, p)
            overlap_len += p_len

        current = overlap_paras
        current_len = sum(len(p) + 2 for p in current)

    for para in paragraphs:
        para_len = len(para) + 2
        if current and current_len + para_len > chunk_size:
            flush_current()
        current.append(para)
        current_len += para_len

    if current:
        chunk_text = "\n\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks

from __future__ import annotations

import re


def lexical_overlap_score(query: str, chunk_text: str) -> float:
    """
    Generic lexical overlap between query and chunk (no domain-specific rules).
    Complements semantic similarity for any document type.
    """
    query_tokens = {t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3}
    if not query_tokens:
        return 0.0

    text_lower = chunk_text.lower()
    matches = sum(1 for t in query_tokens if t in text_lower)
    return min(0.25, (matches / len(query_tokens)) * 0.25)

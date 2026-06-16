from app.core.chunking import split_into_overlapping_chunks


def test_chunking_produces_overlapping_chunks():
    text = "Paragraph one.\n\nParagraph two with more words.\n\nParagraph three."
    chunks = split_into_overlapping_chunks(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_chunking_empty_returns_empty():
    assert split_into_overlapping_chunks("") == []
    assert split_into_overlapping_chunks("   ") == []


def test_chunking_splits_dense_single_paragraph():
    text = "\n".join(f"Line {i} with some content." for i in range(30))
    chunks = split_into_overlapping_chunks(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 3
    assert all(len(c) > 0 for c in chunks)

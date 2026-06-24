from app.core.text import split_chunks


def test_split_chunks():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = split_chunks(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) >= 2


def test_split_empty():
    assert split_chunks("") == []
    assert split_chunks("   ") == []

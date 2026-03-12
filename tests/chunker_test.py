from src.chunker import chunk_text


def test_chunk_text_returns_list():
    text = "Patient Name: John Doe. Diagnosis: Diabetes. " * 20
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert isinstance(chunks, list)


def test_chunk_text_returns_non_empty_chunks():
    text = "Patient Name: John Doe. Diagnosis: Diabetes. " * 20
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk.strip()) > 0 for chunk in chunks)


def test_chunk_text_invalid_overlap():
    text = "Some sample text"
    try:
        chunk_text(text, chunk_size=50, overlap=50)
        assert False, "Expected ValueError when overlap >= chunk_size"
    except ValueError:
        assert True
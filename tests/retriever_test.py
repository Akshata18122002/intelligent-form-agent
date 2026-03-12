import numpy as np
from src.retriever import Retriever


def test_retriever_returns_results():
    embeddings = np.random.rand(3, 384).astype("float32")
    chunks = [
        "Patient Name: John Doe",
        "Diagnosis: Diabetes",
        "Insurance Provider: ABC Health"
    ]
    metadata = [
        {"file_name": "form1.pdf", "chunk_id": 0},
        {"file_name": "form1.pdf", "chunk_id": 1},
        {"file_name": "form1.pdf", "chunk_id": 2},
    ]

    retriever = Retriever(embeddings, chunks, metadata)

    query_embedding = np.random.rand(1, 384).astype("float32")
    results = retriever.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert isinstance(results[0][0], str)
    assert isinstance(results[0][1], dict)
    assert isinstance(results[0][2], float)


def test_retriever_raises_on_empty_embeddings():
    chunks = []
    metadata = []
    embeddings = np.array([], dtype="float32")

    try:
        Retriever(embeddings, chunks, metadata)
        assert False, "Expected ValueError for empty embeddings"
    except ValueError:
        assert True
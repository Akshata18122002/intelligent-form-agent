from src.embedder import Embedder


def test_embedder_returns_embeddings():
    embedder = Embedder()

    texts = [
        "Patient Name: John Doe",
        "Diagnosis: Diabetes"
    ]

    embeddings = embedder.encode_texts(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_embedder_returns_query_embedding():
    embedder = Embedder()

    query_embedding = embedder.encode_query("What is the diagnosis?")

    assert query_embedding.shape[0] == 1
    assert query_embedding.shape[1] > 0
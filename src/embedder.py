from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text chunks into embedding.
        """
        if not texts:
            return np.array([], dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert a single query into an embedding.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.astype("float32")


if __name__ == "__main__":
    sample_texts = [
        "Patient Name: John Doe. Diagnosis: Diabetes.",
        "Insurance claim filed for hospital treatment.",
        "Follow-up appointment scheduled next week."
    ]

    embedder = Embedder()
    embeddings = embedder.encode_texts(sample_texts)

    print("Embeddings shape:", embeddings.shape)

    query_embedding = embedder.encode_query("What is the diagnosis?")
    print("Query embedding shape:", query_embedding.shape)
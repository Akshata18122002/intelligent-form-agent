from typing import List, Tuple
import faiss
import numpy as np


class Retriever:
    def __init__(self, embeddings: np.ndarray, chunks: List[str], metadata: List[dict]):
        """
        Build a FAISS index over chunk embeddings.

        Args:
            embeddings: NumPy array of shape (n_chunks, embedding_dim)
            chunks: List of chunk texts
            metadata: List of metadata dicts, one per chunk
        """
        if len(embeddings) == 0:
            raise ValueError("Embeddings are empty.")

        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length.")

        if len(chunks) != len(embeddings):
            raise ValueError("Number of embeddings must match number of chunks.")

        self.chunks = chunks
        self.metadata = metadata

        self.embeddings = embeddings.astype("float32").copy()
        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, dict, float]]:
        """
        Retrieve top_k most relevant chunks.

        Args:
            query_embedding: NumPy array of shape (1, embedding_dim)
            top_k: Number of results to return

        Returns:
            List of tuples: (chunk_text, metadata, score)
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = query_embedding.astype("float32").copy()
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], self.metadata[idx], float(score)))

        return results


if __name__ == "__main__":
    sample_chunks = [
        "Patient Name: John Doe. Diagnosis: Diabetes.",
        "Insurance claim amount is 5000 dollars.",
        "Follow-up appointment is scheduled next week."
    ]

    sample_metadata = [
        {"file_name": "form1.pdf", "chunk_id": 0},
        {"file_name": "form1.pdf", "chunk_id": 1},
        {"file_name": "form1.pdf", "chunk_id": 2}
    ]

    # dummy embeddings for testing shape flow only
    sample_embeddings = np.random.rand(3, 384).astype("float32")
    sample_query = np.random.rand(1, 384).astype("float32")

    retriever = Retriever(sample_embeddings, sample_chunks, sample_metadata)
    results = retriever.search(sample_query, top_k=2)

    for i, (chunk, meta, score) in enumerate(results, start=1):
        print(f"Result {i}")
        print("Score:", score)
        print("Metadata:", meta)
        print("Chunk:", chunk)
        print("-" * 50)
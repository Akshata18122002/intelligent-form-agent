from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: Input text to split.
        chunk_size: Maximum size of each chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text = text.strip()
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample_text = (
        "Patient Name: John Doe. Age: 45. Diagnosis: Diabetes. "
        "Medication: Metformin. Follow-up in two weeks. "
    ) * 20

    print("=== RUNNING CHUNKER TEST ===")
    chunks = chunk_text(sample_text, chunk_size=120, overlap=30)

    print(f"Total chunks: {len(chunks)}\n", flush=True)
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}:", flush=True)
        print(chunk, flush=True)
        print("-" * 50, flush=True)

    print("=== END OF CHUNKER OUTPUT ===", flush=True)
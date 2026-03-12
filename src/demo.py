from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractor import extract_text_from_pdf
from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.llm import LLMClient
from src.qa_agent import QAAgent
from src.summarizer import Summarizer
from src.multi_form_agent import MultiFormAgent
from src.form_utils import looks_unfilled_template


FORM_DIR = Path("data/forms")


def load_all_forms():
    forms = []

    for pdf_file in FORM_DIR.glob("*.pdf"):
        text = extract_text_from_pdf(str(pdf_file))

        forms.append(
            {
                "file_name": pdf_file.name,
                "text": text,
                "is_unfilled": looks_unfilled_template(text),
            }
        )

    return forms


def build_retriever_for_form(form_text: str, file_name: str, embedder: Embedder) -> Retriever:
    chunks = chunk_text(form_text, chunk_size=500, overlap=100)
    if not chunks:
        raise ValueError(f"No retrievable content for form: {file_name}")

    metadata = [{"file_name": file_name, "chunk_id": i} for i in range(len(chunks))]
    embeddings = embedder.encode_texts(chunks)

    return Retriever(embeddings, chunks, metadata)


def single_form_demo(forms, embedder, qa_agent, summarizer):
    form = forms[0]

    print("\n" + "=" * 80)
    print(f"SINGLE FORM DEMO: {form['file_name']}")
    print("=" * 80)

    if form.get("is_unfilled"):
        print("This form looks like an unfilled template (labels only). Skipping QA and summary.\n")
        return

    print("\n--- SUMMARY ---")
    summary = summarizer.summarize_form(form["text"])
    print(summary)

    retriever = build_retriever_for_form(form["text"], form["file_name"], embedder)

    question = "What symptoms, conditions, or medical issues are explicitly mentioned or confirmed in this form?"
    query_embedding = embedder.encode_query(question)
    results = retriever.search(query_embedding, top_k=3)
    retrieved_chunks = [chunk for chunk, _, _ in results]

    print("\n--- QUESTION ANSWERING ---")
    print("Question:", question)
    print("Answer:", qa_agent.answer_question(question, retrieved_chunks))


def multi_form_demo(forms, embedder, multi_form_agent):
    print("\n" + "=" * 80)
    print("MULTI-FORM DEMO")
    print("=" * 80)

    all_chunks = []
    all_metadata = []

    for form in forms:
        if form.get("is_unfilled"):
            continue

        chunks = chunk_text(form["text"], chunk_size=500, overlap=100)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "file_name": form["file_name"],
                    "chunk_id": i,
                }
            )

    if not all_chunks:
        print("No non-empty forms found for multi-form analysis.")
        return

    embeddings = embedder.encode_texts(all_chunks)
    retriever = Retriever(embeddings, all_chunks, all_metadata)

    question = "Across all forms, what diagnoses, confirmed conditions, or claim-related issues are mentioned?"
    query_embedding = embedder.encode_query(question)
    results = retriever.search(query_embedding, top_k=5)

    contexts = []
    for chunk, meta, score in results:
        contexts.append(f"Form: {meta['file_name']}\n{chunk}")

    print("\nQuestion:", question)
    print("Answer:", multi_form_agent.answer_across_forms(question, contexts))


def main():
    forms = load_all_forms()

    if not forms:
        print("No PDF forms found in data/forms")
        return

    print("Loaded forms:")
    for form in forms:
        print("-", form["file_name"])

    embedder = Embedder()
    llm_client = LLMClient()
    qa_agent = QAAgent(llm_client)
    summarizer = Summarizer(llm_client)
    multi_form_agent = MultiFormAgent(llm_client)

    single_form_demo(forms, embedder, qa_agent, summarizer)
    multi_form_demo(forms, embedder, multi_form_agent)


if __name__ == "__main__":
    main()
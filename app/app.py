from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractor import extract_text_from_pdf
from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.qa_agent import QAAgent
from src.summarizer import Summarizer
from src.llm import LLMClient
from src.multi_form_agent import MultiFormAgent
from src.form_utils import looks_unfilled_template


st.set_page_config(
    page_title="Intelligent Form Agent",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Intelligent Form Agent")
st.caption("Read, extract, summarize, and query one or more forms using a RAG pipeline.")


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_llm_client() -> LLMClient:
    return LLMClient()


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def build_form_record(file_name: str, text: str, embedder: Embedder, chunk_size: int, overlap: int) -> Dict:
    is_unfilled = looks_unfilled_template(text)
    if is_unfilled:
        return {
            "file_name": file_name,
            "text": text,
            "chunks": [],
            "metadata": [],
            "embeddings": [],
            "retriever": None,
            "is_unfilled": True,
        }

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    metadata = [{"file_name": file_name, "chunk_id": i} for i in range(len(chunks))]
    embeddings = embedder.encode_texts(chunks)

    retriever = None
    if len(chunks) > 0 and len(embeddings) > 0:
        retriever = Retriever(embeddings, chunks, metadata)

    return {
        "file_name": file_name,
        "text": text,
        "chunks": chunks,
        "metadata": metadata,
        "embeddings": embeddings,
        "retriever": retriever,
        "is_unfilled": False,
    }


def process_uploaded_files(uploaded_files, embedder: Embedder, chunk_size: int, overlap: int) -> Dict[str, Dict]:
    processed = {}

    for uploaded_file in uploaded_files:
        temp_path = save_uploaded_file(uploaded_file)
        text = extract_text_from_pdf(temp_path)

        processed[uploaded_file.name] = build_form_record(
            file_name=uploaded_file.name,
            text=text,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    return processed


def build_multi_form_retriever(forms_data: Dict[str, Dict]) -> Retriever | None:
    all_chunks: List[str] = []
    all_metadata: List[dict] = []

    for form in forms_data.values():
        for chunk, meta in zip(form["chunks"], form["metadata"]):
            all_chunks.append(chunk)
            all_metadata.append(meta)

    if not all_chunks:
        return None

    embedder = get_embedder()
    all_embeddings = embedder.encode_texts(all_chunks)
    return Retriever(all_embeddings, all_chunks, all_metadata)


def get_single_form_answer(
    question: str,
    form_data: Dict,
    embedder: Embedder,
    qa_agent: QAAgent,
    top_k: int,
):
    retriever = form_data["retriever"]
    if retriever is None:
        return "No retrievable content found in this form.", []

    query_embedding = embedder.encode_query(question)
    results = retriever.search(query_embedding, top_k=top_k)
    retrieved_chunks = [chunk for chunk, _, _ in results]
    answer = qa_agent.answer_question(question, retrieved_chunks)
    return answer, results


def get_multi_form_answer(
    question: str,
    forms_data: Dict[str, Dict],
    embedder: Embedder,
    multi_form_agent: MultiFormAgent,
    top_k: int,
):
    multi_retriever = build_multi_form_retriever(forms_data)
    if multi_retriever is None:
        return "No retrievable content found across uploaded forms.", []

    query_embedding = embedder.encode_query(question)
    results = multi_retriever.search(query_embedding, top_k=top_k)

    contexts = []
    for chunk, meta, _score in results:
        contexts.append(f"Form: {meta['file_name']}\n{chunk}")

    answer = multi_form_agent.answer_across_forms(question, contexts)
    return answer, results


with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1000, value=500, step=50)
    overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=100, step=10)
    top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=8, value=3, step=1)

    st.markdown("---")
    st.write("Recommended:")
    st.write("- Chunk size: 400–600")
    st.write("- Overlap: 80–120")
    st.write("- Top-k: 3–5")

uploaded_files = st.file_uploader(
    "Upload one or more PDF forms",
    type=["pdf"],
    accept_multiple_files=True,
)

if "forms_data" not in st.session_state:
    st.session_state.forms_data = {}

if uploaded_files:
    if st.button("Process uploaded forms"):
        with st.spinner("Extracting text, chunking, and building retrieval index..."):
            embedder = get_embedder()
            st.session_state.forms_data = process_uploaded_files(
                uploaded_files=uploaded_files,
                embedder=embedder,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        st.success(f"Processed {len(st.session_state.forms_data)} form(s).")

forms_data = st.session_state.forms_data

if forms_data:
    llm_client = get_llm_client()
    embedder = get_embedder()
    qa_agent = QAAgent(llm_client)
    summarizer = Summarizer(llm_client)
    multi_form_agent = MultiFormAgent(llm_client)

    form_names = list(forms_data.keys())

    st.subheader("Processed Forms")
    for form_name in form_names:
        form = forms_data[form_name]
        if form.get("is_unfilled"):
            st.warning(f"**{form_name}** looks like an unfilled template (labels only).")
        st.write(
            f"**{form_name}** — extracted characters: {len(form['text'])}, "
            f"chunks: {len(form['chunks'])}"
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single Form QA", "Form Summary", "Multi-Form Analysis", "Inspect Extracted Text"]
    )

    with tab1:
        st.markdown("### Ask a question about one form")
        selected_form = st.selectbox("Choose a form", form_names, key="single_form_select")
        single_question = st.text_input(
            "Question for selected form",
            value="What symptoms, conditions, or medical issues are explicitly mentioned or confirmed in this form?",
        )

        if st.button("Get single-form answer"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                answer, results = get_single_form_answer(
                    question=single_question,
                    form_data=forms_data[selected_form],
                    embedder=embedder,
                    qa_agent=qa_agent,
                    top_k=top_k,
                )

            st.markdown("#### Answer")
            st.success(answer)
            st.caption("Checklist note: unmarked Y/N options are treated as possible fields, not confirmed findings.")

            st.markdown("#### Retrieved Context")
            for i, (chunk, meta, score) in enumerate(results, start=1):
                with st.expander(f"Chunk {i} | Score: {score:.4f} | {meta['file_name']}"):
                    st.write(chunk)

    with tab2:
        st.markdown("### Summarize one form")
        selected_summary_form = st.selectbox("Choose a form to summarize", form_names, key="summary_form_select")

        if st.button("Generate summary"):
            with st.spinner("Generating summary..."):
                summary = summarizer.summarize_form(forms_data[selected_summary_form]["text"])

            st.markdown("#### Summary")
            st.info(summary)
            st.caption("Summary note: checklist options are only treated as present if explicitly marked.")

    with tab3:
        st.markdown("### Ask a question across all uploaded forms")
        multi_question = st.text_input(
            "Cross-form question",
            value="Across all forms, what diagnoses, confirmed conditions, or claim-related issues are mentioned?",
        )

        if st.button("Get multi-form answer"):
            with st.spinner("Retrieving across forms and generating answer..."):
                answer, results = get_multi_form_answer(
                    question=multi_question,
                    forms_data=forms_data,
                    embedder=embedder,
                    multi_form_agent=multi_form_agent,
                    top_k=max(top_k, 4),
                )

            st.markdown("#### Answer")
            st.success(answer)
            st.caption("Checklist note: unmarked Y/N options are treated as possible fields, not confirmed findings.")

            st.markdown("#### Retrieved Chunks Across Forms")
            for i, (chunk, meta, score) in enumerate(results, start=1):
                with st.expander(f"Chunk {i} | Score: {score:.4f} | {meta['file_name']}"):
                    st.write(chunk)

    with tab4:
        st.markdown("### Inspect extracted text")
        inspect_form = st.selectbox("Choose a form to inspect", form_names, key="inspect_form_select")
        st.text_area(
            "Extracted text preview",
            value=forms_data[inspect_form]["text"][:5000],
            height=400,
        )
else:
    st.info("Upload PDF forms and click 'Process uploaded forms' to begin.")
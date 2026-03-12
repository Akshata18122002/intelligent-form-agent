# System Architecture

## Overview

The Intelligent Form Agent follows a **Retrieval-Augmented Generation (RAG)** architecture to process and understand PDF forms. The system combines document processing, semantic retrieval, and large language models to generate answers, summaries, and insights from form data.

The architecture allows the system to handle both **single-form reasoning** and **multi-form analysis**.

---

## Pipeline Overview

The processing pipeline is composed of the following stages:

PDF Forms
↓
Text Extraction
↓
Text Chunking
↓
Embedding Generation
↓
Vector Indexing
↓
Semantic Retrieval
↓
Language Model Reasoning
↓
Answers / Summaries / Insights

---

## Components

### 1. Text Extraction

PDF forms are parsed using **pdfplumber**.

The extractor reads each page and extracts textual content from the form.

Responsibilities:

* open PDF documents
* extract page text
* combine page text into a single document

File:

```
src/extractor.py
```

---

### 2. Text Chunking

Extracted text is divided into smaller overlapping chunks.

Chunking improves retrieval performance because the language model receives only the most relevant sections of the document.

Responsibilities:

* split text into manageable segments
* maintain overlap between chunks
* preserve semantic context

File:

```
src/chunker.py
```

---

### 3. Embedding Generation

Each chunk is converted into a vector embedding using a **Sentence Transformer model**.

Embeddings capture the semantic meaning of text and allow similarity search.

Responsibilities:

* convert chunks to vectors
* convert user queries to vectors

File:

```
src/embedder.py
```

---

### 4. Vector Indexing and Retrieval

Embeddings are indexed using **FAISS**.

When a user asks a question, the system retrieves the most relevant chunks using similarity search.

Responsibilities:

* build vector index
* perform nearest-neighbor search
* return relevant document chunks

File:

```
src/retriever.py
```

---

### 5. LLM Reasoning

The retrieved chunks are passed to a language model to generate responses.

The LLM is responsible for:

* answering questions
* summarizing forms
* reasoning across multiple forms

File:

```
src/llm.py
```

---

### 6. Agents

Three agents are implemented on top of the LLM:

#### Question Answering Agent

Answers questions about a single form.

```
src/qa_agent.py
```

#### Summarization Agent

Generates concise summaries highlighting important details.

```
src/summarizer.py
```

#### Multi-Form Analysis Agent

Combines information from multiple forms to produce higher-level insights.

```
src/multi_form_agent.py
```

All agents are **checklist-aware** for medical forms that use Y/N-style options:

- They only treat a symptom or condition as present when the corresponding checkbox is explicitly marked (e.g. ✓, ✔, ☑, [x]/[X]).
- Unmarked options such as `□Y□N Fever` are treated as *possible fields*, not confirmed findings.

---

## End-to-End Flow

The complete system flow is as follows:

1. Load PDF forms
2. Extract text from documents
3. Split text into chunks
4. Generate embeddings
5. Index embeddings in FAISS
6. Retrieve relevant chunks for a query
7. Send retrieved context to the language model
8. Generate answers, summaries, or insights

---

## Design Advantages

The architecture offers several benefits:

* efficient retrieval using vector search
* scalable analysis across many documents
* modular pipeline design
* ability to integrate more advanced models in the future

---

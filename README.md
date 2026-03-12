# Intelligent Form Agent – Read, Extract, and Explain

## Overview

The **Intelligent Form Agent** is a document understanding system that processes and analyzes structured and semi-structured forms (such as medical intake forms, insurance claim forms, or registration forms).

The agent can:

* Extract text from PDF forms
* Answer questions about individual forms
* Generate concise summaries of forms
* Provide holistic insights across multiple forms

The system is built using a **Retrieval-Augmented Generation (RAG)** pipeline that combines semantic embeddings, vector search, and a language model to understand and reason about form content.

---

## Features

### 1. PDF Text Extraction

The system extracts text from PDF forms using `pdfplumber`.

### 2. Question Answering

Users can ask questions about individual forms.

Example query:

```
What symptoms, conditions, or medical issues are explicitly mentioned or confirmed in this form?
```

### 3. Form Summarization

The agent generates concise summaries highlighting the most important information contained in a form.

### 4. Multi-Form Insights

The system can analyze multiple forms together to identify common themes or issues.

Example query:

```
Across all forms, what diagnoses, confirmed conditions, or claim-related issues are mentioned?
```

### 5. Semantic Retrieval

The system uses embeddings and FAISS to retrieve the most relevant sections of the form before sending them to the language model.

### 6. Checklist-Aware Medical Reasoning

Many medical PDFs include symptom checklists (often with Y/N boxes). This project is **checkbox-aware**:

- Only treats symptoms/conditions as **present** when an item is explicitly marked with **✓ / ✔ / ☑ / [x]**.
- Unmarked Y/N options (e.g., `□Y□N Fever`) are treated as **possible fields**, not confirmed findings.

---

## Architecture

The system follows a **Retrieval-Augmented Generation pipeline**:

```
PDF Forms
   ↓
Text Extraction (pdfplumber)
   ↓
Text Chunking
   ↓
Embedding Generation (Sentence Transformers)
   ↓
Vector Search (FAISS)
   ↓
Relevant Context Retrieval
   ↓
LLM Reasoning
   ↓
Answers, Summaries, Insights
```

---

## Project Structure

```
intelligent-form-agent
│
├── app
│   └── app.py
│
├── src
│   ├── extractor.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── llm.py
│   ├── form_utils.py
│   ├── qa_agent.py
│   ├── summarizer.py
│   ├── multi_form_agent.py
│   └── demo.py
│
├── data
│   └── forms
│       ├── form1.pdf
│       ├── form2.pdf
│       └── form3.pdf
│
├── notebooks
│   └── experiments.ipynb
│
├── tests
│   ├── test_extractor.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_retriever.py
│
├── docs
│   └── architecture.md
│
├── requirements.txt
├── README.md
└── .env
```

---

## Setup Instructions

### 1. Create Virtual Environment

```
python -m venv venv
```

Activate it:

Windows

```
.\venv\Scripts\Activate.ps1
```

Mac / Linux

```
source venv/bin/activate
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Add API Key

Create a `.env` file in the project root.

Example:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

### 4. Add Sample Forms

Place sample forms inside:

```
data/forms/
```

Example:

```
data/forms/form1.pdf
data/forms/form2.pdf
data/forms/form3.pdf
```

---

## Running the Agent

### Option A: Run the Streamlit UI (recommended)

```bash
streamlit run .\app\app.py
```

The UI supports:

- Uploading one or more PDFs
- Processing (extract → chunk → embed → index)
- Single-form QA
- Per-form summary
- Multi-form analysis
- Inspecting extracted text

Notes:

- If your PDFs contain checklist sections, the UI treats only **explicitly checked** items as present.
- You may see a small caption under answers clarifying checklist interpretation.

### Option B: Run the CLI demo

```
python -m src.demo
```

This will demonstrate:

* form summarization
* question answering
* multi-form analysis

---

## Example Runs

### Question Answering

Example query:

```
What symptoms, conditions, or medical issues are explicitly mentioned or confirmed in this form?
```

Example output:

```
Respiratory symptoms such as coughing, shortness of breath,
and wheezing are mentioned in the form.
```

---

### Form Summary

Example output:

```
• Patient questionnaire form
• Symptoms reported include fatigue and fever
• Respiratory and cardiovascular issues mentioned
• Medication and allergy information included
```

---

### Multi-Form Insights

Example query:

```
Across all forms, what diagnoses, confirmed conditions, or claim-related issues are mentioned?
```

Example output:

```
Respiratory symptoms, cardiovascular complaints,
and gastrointestinal issues appear across multiple forms.
```

---

## Running Tests

Run unit tests using:

```
pytest
```

Tests cover:

* PDF text extraction
* text chunking
* embedding generation
* semantic retrieval

---

## Creative Extensions

Possible improvements include:

* OCR support for scanned forms
* layout-aware document models (e.g. LayoutLM)
* structured field extraction
* richer UI for uploading and querying forms (authentication, saved sessions, export)
* visualization of multi-form insights

---

## Conclusion

The Intelligent Form Agent demonstrates how modern NLP techniques and LLM-based reasoning can automate document understanding tasks.

By combining semantic retrieval with language models, the system can extract information from forms, answer questions, generate summaries, and provide insights across multiple documents.

This approach significantly reduces manual review effort and enables scalable document analysis workflows.

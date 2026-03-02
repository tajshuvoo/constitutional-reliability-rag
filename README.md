# Constitutional Reliability RAG

**Live Application:**  
https://constitutional-reliability-rag.streamlit.app/

**Source Code:**  
https://github.com/tajshuvoo/constitutional-reliability-rag  

---

## Overview

This project implements a **Reliability-Aware Agentic Retrieval-Augmented Generation (RAG)** system for question answering over the Constitution of Bangladesh.

Unlike standard RAG pipelines, this system enforces:

- Citation grounding  
- LLM-based faithfulness verification  
- Automatic self-correction loop  
- Structured reliability flags  
- Quantitative benchmarking  

The system is designed to move beyond demo-style QA and focus on measurable reliability.

---

## Architecture

```
User Query
    ↓
Hybrid Retrieval (FAISS + semantic search)
    ↓
LLM Generation with strict citation rules
    ↓
Evaluation Layer
    • Citation validation
    • Faithfulness verification (LLM-based)
    ↓
Self-correction (if unreliable)
    ↓
Final answer with reliability metadata
```

---

## Key Features

- Agent orchestration using LangGraph  
- Structured state management via Pydantic  
- JSON-validated LLM outputs  
- Citation enforcement with regex-based validation  
- Faithfulness classification (SUPPORTED / UNSUPPORTED)  
- Automatic retry and correction mechanism  
- Streaming FastAPI backend  
- Streamlit frontend with reliability dashboard  
- Benchmark harness for quantitative evaluation  

---

## Benchmarking

Includes a structured evaluation pipeline:

```
run_benchmark.py
```

### Evaluation Metrics

- Reliability rate  
- Retrieval recall rate  
- Correct refusal accuracy  
- Correction trigger rate  
- Correction success rate  

This enables systematic reliability improvement instead of qualitative inspection.

---

## Tech Stack

- Python  
- FastAPI  
- LangGraph  
- LangChain  
- FAISS  
- HuggingFace Inference API  
- Streamlit  
- Pydantic  

---

## Project Structure

```
backend/
│
├── app/
│   ├── agent/
│   ├── evaluation/
│   ├── retrieval/
│   ├── services/
│   ├── schemas/
│   └── main.py
│
└── data/

frontend/
└── app.py
```

---


## Run Locally

```bash
uv sync
uv run uvicorn backend.app.main:app --reload
uv run streamlit run frontend/app.py
```

---

## Author

**Md. Tajbir Hasan Shuvo**  
CSE, Rajshahi University of Engineering & Technology  

GitHub: https://github.com/tajshuvoo  
LinkedIn: https://www.linkedin.com/in/tajshuvo
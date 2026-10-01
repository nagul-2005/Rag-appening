# 🎯 RAG Project — Interview Walkthrough & Explanation Guide

This guide is written in plain, natural language so you can comfortably explain your project to an interviewer without sounding robotic or reciting canned script.

---

## 💡 How to Describe the Project in 30 Seconds

> *"I built a grounded RAG chatbot for the 'Agentic AI eBook' using Python, LangGraph, FastAPI, ChromaDB, and Streamlit.*
> *Instead of a basic LangChain chain where whatever gets retrieved is blindly sent to an LLM, I structured it as a 3-step LangGraph workflow: **Retrieve ➔ Grade ➔ Generate**.*
> *This lets us inspect retrieved chunks, calculate a confidence score, and stop the model from hallucinating if the eBook doesn't cover the topic."*

---

## 🧱 The Architecture in Simple Terms

```
User Query (from Streamlit UI)
      │
      ▼
FastAPI Server (POST /chat)
      │
      ▼
LangGraph Pipeline:
  1. [retrieve]        -> Searches ChromaDB for top 4 chunks
  2. [grade_documents] -> Filters out chunks below 0.25 similarity; computes confidence score
  3. [generate]        -> Uses Gemini 2.5 Flash (temperature 0) to write a grounded answer
      │
      ▼
JSON Response with answer, cited chunks (page numbers), and confidence score
```

---

## 🗣️ How to Walk Through the Files Line by Line

If the interviewer asks: **"Show me your code structure"**, explain it in this order:

### 1. `src/config.py` — Settings
- **What it does**: Central place for settings (chunk size, model names, file paths).
- **How to explain it**:
  > *"I used `pydantic-settings` so variables are automatically read from `.env`. This keeps API keys and file paths out of the source code."*

### 2. `src/utils.py` — Math & Helper Functions
- **`download_pdf()`**: Downloads the eBook if not already in `data/`.
- **`distance_to_score(distance)`**:
  > *"ChromaDB returns distance (0 means exact match, bigger means less similar). I convert it to a 0–1 similarity score using `1 / (1 + distance)`."*
- **`compute_confidence(scores)`**:
  > *"I calculate an overall confidence score giving 60% weight to the top matching chunk and 40% to the average of all chunks."*

### 3. `src/ingest.py` — Preprocessing the PDF
- **What it does**:
  1. Downloads the PDF.
  2. Loads pages with `PyPDFLoader`.
  3. Splits them into chunks of 800 characters with 150 overlap using `RecursiveCharacterTextSplitter`.
  4. Embeds chunks with Google's `gemini-embedding-001` and saves them to local ChromaDB.
- **Why 800 size and 150 overlap?**:
  > *"800 characters is about 120–150 words — long enough to hold a complete idea, but short enough to avoid irrelevant text. The 150-character overlap prevents sentences from getting cut in half across chunk boundaries."*

### 4. `src/graph.py` — The Core LangGraph Workflow
This is the heart of the project. Explain the 3 nodes:
- **Node 1: `retrieve`**:
  > *"Takes the user question, queries ChromaDB for the top 4 chunks, and normalizes the distance to a 0–1 similarity score."*
- **Node 2: `grade_documents`**:
  > *"Filters out any chunk whose score is below 0.25. If no chunks pass, we know the document doesn't have the answer."*
- **Node 3: `generate`**:
  > *"If no chunks survived grading, it immediately returns: 'I cannot answer this question based on the provided eBook context'. If chunks did pass, it feeds them to Gemini 2.5 Flash with temperature 0.0 and a strict prompt forbidding outside knowledge."*
- **Why LangGraph instead of a simple chain?**:
  > *"With LangGraph, each step is an isolated, testable function. You can easily add loops, query rewrites, or human approval without rebuilding the pipeline."*

### 5. `main.py` — The FastAPI Backend
- **`POST /chat`**: Receives `{"question": "..."}`, calls `run_rag_pipeline()`, and returns `{"answer": "...", "retrieved_chunks": [...], "confidence_score": 0.78}`.
- **`GET /health`**: Returns system status and the count of indexed chunks in ChromaDB.
- **Why FastAPI?**:
  > *"It's fast, asynchronous, validates requests using Pydantic, and gives you automatic interactive documentation at `/docs`."*

### 6. `app.py` — Streamlit UI
- **What it does**: Clean chat UI with message history, clickable sample questions, confidence score badge, and an expander to inspect the actual source chunks and page numbers.
- **How it connects**:
  > *"It calls the FastAPI `/chat` endpoint over HTTP, but if the FastAPI server is stopped, it gracefully falls back to calling the LangGraph pipeline directly."*

---

## 🙋 Likely Interview Questions & Direct Answers

### Q1: "How do you avoid hallucinations?"
> *"Two ways: First, the grading node discards low-similarity chunks and short-circuits if nothing relevant was found. Second, we set LLM temperature to 0.0 and strictly instruct it in the system prompt to only use the provided context snippets."*

### Q2: "What if the user asks something completely off-topic?"
> *"ChromaDB returns chunks with low similarity. The grading node filters them out, marks confidence as 0.0, and returns a clear refusal without making an unnecessary LLM call."*

### Q3: "Why ChromaDB instead of Pinecone?"
> *"For this assignment, ChromaDB is lightweight, runs embedded in Python, and persists directly to disk in `data/chroma_db` without needing cloud infrastructure or API keys. If we scaled to millions of documents across multiple servers, we'd switch to Pinecone or Qdrant."*

### Q4: "How would you improve this in production?"
> *"1. Add a **Query Rewriter** node before retrieval to expand abbreviations and improve keyword matching.*
> *2. Add a **Cross-Encoder Reranker** (like Cohere Rerank) to re-order the retrieved chunks for higher precision.*
> *3. Add a **semantic cache** (like Redis) so frequent queries return instantly without re-calling the LLM."*

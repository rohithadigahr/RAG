# RAG
Single Rag application using standard dataset available on Kaggle , FAISS for vector store and similarity search and Ollama for response generation.

# Retrieval-Augmented Generation (RAG) Pipeline with TinyLlama via Ollama

This project implements a lightweight **RAG (Retrieval-Augmented Generation)** pipeline using:
* **Sentence Transformers** for document embeddings
* **FAISS** for similarity search
* **TinyLlama (via Ollama)** for response generation
* **FastAPI** to serve the RAG pipeline

## Folder Structure

```
rag/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app to serve RAG API
│   └── rag_pipeline.py              # Core RAG logic (retrieval + generation)
├── data/
│   └── synthetic_knowledge_items.csv # Sample knowledge base
├── index/
│   ├── faiss.index                  # FAISS vector index
│   └── doc_metadata.csv             # Original document metadata for retrieval
├── prepare_data.py                  # Preprocess and build FAISS index
├── requirements.txt                 # Python dependencies
├── run.py                           # Script to run FastAPI with uvicorn
└── README.md                        # Project documentation
```

## 🛠️ Setup Instructions

### 1. 🔁 Clone the Repository

```bash
git clone <your-repo-url>
cd rag
```

### 2. 🐍 Create Virtual Environment

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Step 1: Prepare the FAISS Index

1. Make sure your data exists in `data/synthetic_knowledge_items.csv`
2. Run the indexing script:

```bash
python prepare_data.py
```

This will create:
* `index/faiss.index`: Vector index
* `index/doc_metadata.csv`: Document metadata

## Step 2: Run the RAG API

1. Ensure Ollama is installed and running locally
2. Pull the TinyLlama model (if not already pulled):

```bash
ollama pull tinyllama
```

3. Start the model server:

```bash
ollama run tinyllama
```

4. Then, in another terminal, start your FastAPI app:

```bash
python run.py
```

5. Visit: http://127.0.0.1:8000/docs to test the endpoints

## How It Works

1. **User submits a query**
2. `SentenceTransformer` embeds the query
3. FAISS retrieves top-k similar documents from the knowledge base
4. Context is combined and sent to **TinyLlama (via Ollama)** for response generation
5. Response is returned to the user

## Evaluation

The RAG pipeline includes a simple evaluator to measure how well the **generated answer** matches the **retrieved context**, using **cosine similarity** between embeddings:

```python
evaluate_generation_vs_retrieved(generated_answer, context, embedder)
```

## 🧪 API Endpoint Example

### `POST /rag`

**Request body:**
```json
{
  "query": "How do I reset my password?"
}
```

**Response:**
```json
{
  "query": "How do I reset my password?",
  "answer": "Step-by-step instructions based on documentation...",
  "similarity_score": 0.87,
  "context": [...],
  "latency_ms": 21.47623,
  "memory_used_mb": 36.41,
  "retrieval_quality": 0.8394,
}
```

## Models Used

* `all-MiniLM-L6-v2` — Sentence embeddings
* `tinyllama` — Local LLM served via Ollama

## Requirements

* Python 3.9+
* Ollama installed and accessible from terminal
* Internet access for initial model pulls

## Optional Cleanup

To delete existing indexes and regenerate:

```bash
rm -rf index/*
python prepare_data.py
```

## Contact

Maintained by [Rohith Adiga Hr]. Questions, issues, or suggestions? Raise an issue or contact directly.



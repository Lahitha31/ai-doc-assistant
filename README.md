# AI Document Assistant Retrieval-Augmented Generation (RAG) App

This project is a lightweight AI-powered document assistant that allows users to upload their own files (PDF, TXT, Markdown) and ask natural language questions about them. Instead of using a chatbot with general internet knowledge, this system provides **accurate, document-grounded answers** using Retrieval-Augmented Generation (RAG).



---

##  Why I Built This

Most AI chatbots hallucinate or provide vague answers when it comes to private or enterprise data. This project solves that problem by:

- **Indexing custom user documents**
- **Searching using FAISS vector database**
- **Retrieving relevant chunks using embeddings**
- **Generating concise answers using a local language model (FLAN-T5)**

---

##  Key Features

-  **Upload your own files** directly from the browser
-  **One-click reindexing** (no terminal required)
-  **Ask any question and get grounded, source-backed answers**
-  **Uses Retrieval-Augmented Generation (RAG)** for higher accuracy
-  **Supports evaluation and debugging** through a built-in `/debug/search` endpoint


---

## How It Works (Simple Overview)

1. You upload documents through the web interface
2. The system splits them into chunks and converts those chunks into embeddings
3. FAISS stores these embeddings for fast vector similarity search
4. When you ask a question, the system retrieves the most relevant chunks
5. The language model generates a focused answer using only those chunks (reducing hallucinations)

---

## Getting Started (Local Setup)

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate   # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Place documents in the data/ folder or use the web upload
python -m app.ingest     # Builds the search index

# Run the app
uvicorn app.main:app --reload --port 8000
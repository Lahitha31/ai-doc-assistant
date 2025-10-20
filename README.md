\# AI Document Assistant · RAG + FAISS + FastAPI (CPU-Friendly)



A lightweight Retrieval-Augmented Generation (RAG) app that indexes your \*\*PDF/TXT/MD\*\* files into \*\*FAISS\*\* and serves answers via \*\*FastAPI\*\*. CPU-friendly defaults (FLAN-T5) so it runs locally without a GPU.


---



\##Features



\- \*\*Upload \& Reindex\*\* from the browser (no CLI needed)

\- \*\*RAG pipeline\*\*: HuggingFace embeddings + FAISS retriever + T5 generator

\- \*\*Simple Web UI\*\* (one input, one answer box)

\- \*\*Sources shown\*\* under each answer (filename + snippet)

\- \*\*/debug/search\*\* endpoint to inspect what chunks were retrieved

\- \*\*Windows-friendly\*\* setup and commands



---



\## Tech Stack



\- \*\*Backend\*\*: FastAPI, Uvicorn

\- \*\*RAG\*\*: FAISS, sentence-transformers (all-MiniLM-L6-v2), Hugging Face pipelines

\- \*\*Frontend\*\*: Minimal HTML/JS (no framework)

\- \*\*Optional\*\*: RAGAS for eval (already wired in the codebase as a utility)



---



\## Quick Start (Windows / PowerShell)



```powershell

git clone https://github.com/YOUR\_USERNAME/YOUR\_REPO.git

cd YOUR\_REPO



python -m venv .venv

.venv\\Scripts\\Activate



pip install -r requirements.txt

##How to Use



Open http://localhost:8000/



Type a question → Ask



To add docs:



Click Upload \& Reindex (supports multiple files), or



Drop files into ./data/ and click Re-Load Documents



Answers show below with Sources for transparency







##Troubleshooting

Weird characters (’ shows as â€™): use browser UI; PowerShell may show legacy encoding.

Answers from old docs: Click Re-Load Documents or call POST /reload (rebuild index).

Uploads not appearing: Check server logs; supported types: .pdf, .txt, .md.

FAISS or sentence-transformers errors: run
pip install --upgrade pip setuptools wheel
pip install sentence-transformers faiss-cpu

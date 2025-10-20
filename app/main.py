from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from pathlib import Path
from app.ingest import build_index
from app.rag_pipeline import RAGPipeline
import traceback

app = FastAPI(title="AI Document Assistant")

# CORS fix to allow frontend JS to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None

@app.on_event("startup")
async def load_rag():
    global rag
    try:
        rag = RAGPipeline()
        print("✅ RAG Pipeline loaded at startup.")
    except Exception as e:
        print("⚠️ Pipeline not loaded yet:", repr(e))
        rag = None

@app.get("/")
def serve_ui():
    index_file = Path(__file__).with_name("index.html")
    return FileResponse(index_file)

@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": rag is not None}

@app.post("/ingest")
def ingest():
    global rag
    try:
        build_index()
        rag = RAGPipeline()
        return {"status": "indexed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/ask")
def ask(request: dict):
    global rag
    if rag is None:
        return {"error": "Index not loaded. Please click ingest first."}
    
    question = request.get("question", "").strip()
    if not question:
        return {"error": "Please provide a question."}

    try:
        result = rag.answer(question)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/reload")
def reload_index():
    """Rebuild FAISS from existing files in ./data"""
    global rag
    try:
        build_index()
        rag = RAGPipeline()
        return {"status": "indexed"}
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload files to ./data, then rebuild index automatically"""
    allowed = {".pdf", ".txt", ".md"}
    saved = []

    for up in files:
        ext = Path(up.filename).suffix.lower()
        if ext not in allowed:
            return JSONResponse(
                {"status": "error", "error": f"Unsupported file type: {ext}"},
                status_code=400,
            )
        fname = Path(up.filename).name  # sanitize
        dest = DATA_DIR / fname
        with dest.open("wb") as f:
            shutil.copyfileobj(up.file, f)
        saved.append(str(dest))

    try:
        build_index()
        global rag
        rag = RAGPipeline()
        return {"status": "ok", "saved": saved}
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)





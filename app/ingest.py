from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.settings import settings

def load_documents(data_dir):
    docs = []
    for p in Path(data_dir).glob("**/*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            docs += PyPDFLoader(str(p)).load()
        elif ext in {".txt", ".md"}:
            docs += TextLoader(str(p), encoding="utf-8").load()
    return docs

def build_index():
    docs = load_documents(settings.DATA_DIR)
    if not docs:
        raise RuntimeError(f"No docs found in {settings.DATA_DIR}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name=settings.HF_EMBED_MODEL)
    vs = FAISS.from_documents(chunks, emb)
    Path(settings.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(settings.INDEX_DIR)

if __name__ == "__main__":
    build_index()
    print("✅ Index built.")

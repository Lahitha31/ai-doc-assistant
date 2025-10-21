from ftfy import fix_text
from pathlib import Path
from shutil import rmtree
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.settings import settings

def normalize_text(t: str) -> str:
    # fix mojibake (â€™ etc.) and collapse whitespace
    t = fix_text(t)
    return " ".join(t.split())

def load_documents(data_dir):
    docs = []
    data_path = Path(data_dir)
    for p in data_path.glob("**/*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            loaded = PyPDFLoader(str(p)).load()
            for d in loaded:
                d.page_content = normalize_text(d.page_content)
            docs += loaded
        elif ext in {".txt", ".md"}:
            try:
                loaded = TextLoader(str(p), autodetect_encoding=True).load()
            except Exception:
                text = p.read_text(encoding="utf-8", errors="ignore")
                loaded = [Document(page_content=text, metadata={"source": str(p)})]
            for d in loaded:
                d.page_content = normalize_text(d.page_content)
            docs += loaded
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

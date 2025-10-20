from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from app.settings import settings

def get_vectorstore():
    emb = HuggingFaceEmbeddings(model_name=settings.HF_EMBED_MODEL)
    idx_dir = Path(settings.INDEX_DIR)
    vs = FAISS.load_local(str(idx_dir), emb, allow_dangerous_deserialization=True) if idx_dir.exists() else None
    return vs, emb

def build_llm():
    text_gen = pipeline(
        "text2text-generation",
        model=settings.HF_TEXT_GEN_MODEL,
        device=-1,       # Force CPU
        max_new_tokens=128
    )
    return HuggingFacePipeline(pipeline=text_gen)


def get_prompt():
    sys = ("You are a helpful assistant. Use ONLY the provided context. "
           "If the answer is not in the context, say you don't know.")
    return ChatPromptTemplate.from_messages([
        ("system", sys),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

class RAGPipeline:
    def __init__(self):
        self.vs, self.emb = get_vectorstore()
        if self.vs is None:
            raise RuntimeError("No FAISS index found. Run ingest first.")
        self.llm = build_llm()
        self.prompt = get_prompt()

    def retrieve(self, query, k=None):
        k = k or settings.TOP_K
        return self.vs.similarity_search(query, k=k)

    def answer(self, question):
        docs = self.retrieve(question)
        ctx = format_context(docs)
        prompt_text = self.prompt.format(question=question, context=ctx)
        out = self.llm.invoke(prompt_text)

        # build neat sources list
        sources = []
        for d in docs:
            meta = d.metadata or {}
            snippet = d.page_content[:180].replace("\n", " ").strip()
            sources.append({
                "file": meta.get("source", "unknown"),
                "page": meta.get("page", None),
                "snippet": snippet + ("…" if len(d.page_content) > 180 else "")
            })

        return {
            "answer": out,
            "sources": sources,
            "meta": {"k": settings.TOP_K}
        }



from pathlib import Path
from typing import List
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from app.settings import settings


def _format_context(docs) -> str:
    """Compact, de-duplicated context fed to the LLM."""
    parts: List[str] = []
    seen = set()
    for d in docs:
        text = (d.page_content or "").strip()
        # avoid dumping identical chunks
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)
        src = (d.metadata or {}).get("filename") or (d.metadata or {}).get("source") or "unknown"
        page = (d.metadata or {}).get("page")
        tag = f"[{src}{f' p.{page}' if page is not None else ''}]"
        parts.append(f"{tag}\n{text}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    """
    Minimal RAG pipeline:
      - Loads FAISS index with HF embeddings
      - Retrieves with Max Marginal Relevance (MMR) to reduce duplicates
      - Generates concise answers with a text2text LLM (e.g., FLAN-T5)
    """

    def __init__(self):
        # Embeddings + Vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.HF_EMBED_MODEL)

        index_dir = Path(settings.INDEX_DIR)
        if not index_dir.exists():
            raise RuntimeError(
                f"Index not found in {index_dir}. Run ingestion first: `python -m app.ingest`"
            )

        # allow_dangerous_deserialization=True is needed for FAISS load on newer langchain versions
        self.vs = FAISS.load_local(
            str(index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # LLM
        self.llm = self.build_llm()

        # Prompt template (short & anti-repetition)
        self.prompt_template = (
            "You are a helpful assistant. Use ONLY the context below.\n"
            "Answer concisely in 1–3 sentences. Do not repeat lines or phrases.\n"
            "If the answer is not contained in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )


    # LLM
  
    def build_llm(self):
        """
        Build a Hugging Face text-generation pipeline and wrap it for LangChain.
        CPU mode uses device=-1. For GPU, set device=0.
        """
        text_gen = pipeline(
            "text2text-generation",
            model=settings.HF_TEXT_GEN_MODEL,  # e.g., "google/flan-t5-base"
            device=-1,
        )

        # Wrap HF pipeline so we can call llm.invoke(prompt_text)
        llm = HuggingFacePipeline(
            pipeline=text_gen,
            model_kwargs={
                "max_new_tokens": 128,
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
                "early_stopping": True,
            },
        )
        return llm

  
    # Retrieval
  
    def retrieve(self, question: str, k: int | None = None):
        """
        Retrieve with MMR to reduce near-duplicate chunks in the final context.
        """
        k = k or settings.TOP_K
        return self.vs.max_marginal_relevance_search(
            question, k=k, fetch_k=20, lambda_mult=0.6
        )

    
    # Answer
 
    def answer(self, question: str):
        docs = self.retrieve(question)
        context = _format_context(docs)
        prompt_text = self.prompt_template.format(context=context, question=question)

        # HuggingFacePipeline.invoke returns a string
        out = self.llm.invoke(prompt_text)
        answer_text = out if isinstance(out, str) else str(out)

        # Build neat sources list for the UI
        sources = []
        for d in docs:
            meta = d.metadata or {}
            fname = meta.get("filename") or meta.get("source", "unknown")
            page = meta.get("page", None)
            snippet_raw = (d.page_content or "").replace("\n", " ").strip()
            snippet = (snippet_raw[:180] + "…") if len(snippet_raw) > 180 else snippet_raw
            sources.append({"file": fname, "page": page, "snippet": snippet})

        return {
            "answer": answer_text,
            "sources": sources,
            "meta": {"k": settings.TOP_K},
        }

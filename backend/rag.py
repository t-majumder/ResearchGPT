import os
from typing import List, Dict, Any, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    GROQ_API_KEY,
    PDFS_DIRECTORY,
    DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ADD_START_INDEX,
    SEARCH_TYPE,
    TOP_K_RESULTS,
    FETCH_K,
    MIN_SIMILARITY,
    USE_RERANKER,
    RERANKER_MODEL,
    RERANK_TOP_N,
    MAX_HISTORY_MESSAGES,
    PROMPT_BEHAVIOR_WITH_RAG,
    PROMPT_BEHAVIOR_WITHOUT_RAG,
    AVAILABLE_MODELS,
)

AVAILABLE_MODELS = AVAILABLE_MODELS

ROUTING_PROMPT = """
You are a routing assistant. Decide if the question needs document retrieval.
Return ONLY one word: "RETRIEVE" or "DIRECT".

RETRIEVE if it needs PDF content; DIRECT if general/greeting.

Question: {question}
Decision:
"""

PROMPT_WITH_DOCS = f"""{PROMPT_BEHAVIOR_WITH_RAG}""" + """
Chat History:
{chat_history}

Context from PDFs:
{context}

Question: {question}
Answer:
"""

PROMPT_WITHOUT_DOCS = f"""{PROMPT_BEHAVIOR_WITHOUT_RAG}""" + """

Chat History:
{chat_history}

Question: {question}
Answer:
"""


def make_llm(model_name: str):
    model_id = AVAILABLE_MODELS.get(model_name, list(AVAILABLE_MODELS.values())[0])
    if model_id.startswith("ollama:"):
        ollama_model = model_id.split("ollama:")[1]
        return ChatOllama(model=ollama_model)
    return ChatGroq(model=model_id, api_key=GROQ_API_KEY)


def format_chat_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No previous conversation."
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    out = []
    for m in trimmed:
        role = "User" if m.get("role") == "user" else "Assistant"
        out.append(f"{role}: {m.get('content','')}")
    return "\n".join(out)


# ---- CONTEXT PACKING ----
MAX_CONTEXT_CHARS = 14000 

def docs_to_context(docs) -> Tuple[str, List[Dict[str, Any]]]:
    used = []
    parts = []
    total = 0

    for d in docs:
        src = os.path.basename(str(d.metadata.get("source", "unknown")))
        page = d.metadata.get("page", None)
        header = f"[source={src}" + (f", page={page}]" if page is not None else "]")

        chunk = d.page_content.strip()
        if not chunk:
            continue
        chunk = chunk[:2500]

        block = header + "\n" + chunk + "\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total += len(block)

        used.append({
            "source": src,
            "page": page,
            "preview": chunk[:220].replace("\n", " ")
        })

    context = "\n---\n".join(parts).strip()
    return context, used


# ---------------- RERANKER ---------------- #
class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, docs: List, top_n: int) -> List[Tuple[Any, float]]:
        self._load()
        pairs = [(query, d.page_content[:2500]) for d in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return ranked[:top_n]


_reranker = CrossEncoderReranker(RERANKER_MODEL) if USE_RERANKER else None


# ---------------- VECTOR DB ---------------- #
class VectorDB:
    def __init__(self):
        os.makedirs(PDFS_DIRECTORY, exist_ok=True)
        os.makedirs(DB_PATH, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.faiss_db: Optional[FAISS] = None

    def _index_exists(self) -> bool:
        return os.path.exists(os.path.join(DB_PATH, "index.faiss"))

    def load_if_exists(self) -> bool:
        if self.faiss_db is not None:
            return True
        if self._index_exists():
            self.faiss_db = FAISS.load_local(
                DB_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return True
        return False

    def list_pdfs(self) -> List[str]:
        return [f for f in os.listdir(PDFS_DIRECTORY) if f.lower().endswith(".pdf")]

    def pdf_count(self) -> int:
        return len(self.list_pdfs())

    def _load_all_pdfs(self):
        documents = []
        for fn in self.list_pdfs():
            path = os.path.join(PDFS_DIRECTORY, fn)
            loader = PDFPlumberLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = fn
            documents.extend(docs)
        return documents

    def _chunk(self, documents):
        splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=ADD_START_INDEX,
            encoding_name="cl100k_base",
        )
        return splitter.split_documents(documents)

    def rebuild(self) -> Dict[str, Any]:
        documents = self._load_all_pdfs()
        if not documents:
            self.faiss_db = None
            return {"ok": False, "reason": "No PDFs found in pdfs/ directory"}

        chunks = self._chunk(documents)
        if len(chunks) == 0:
            self.faiss_db = None
            return {"ok": False, "reason": "Chunking produced 0 chunks (empty extraction?)"}

        self.faiss_db = FAISS.from_documents(chunks, self.embeddings)
        self.faiss_db.save_local(DB_PATH)
        return {"ok": True, "pdfs": self.pdf_count(), "chunks": len(chunks)}

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        return 1.0 / (1.0 + float(distance))

    def retrieve(self, query: str) -> Tuple[List[Any], Dict[str, Any]]:
        if self.faiss_db is None:
            self.load_if_exists()
        if self.faiss_db is None:
            return [], {"ok": False, "reason": "DB not loaded"}

        debug = {"ok": True, "fetch_k": FETCH_K, "min_similarity": MIN_SIMILARITY}

        docs_scores = self.faiss_db.similarity_search_with_score(query, k=FETCH_K)

        candidates = []
        for doc, dist in docs_scores:
            sim = self._distance_to_similarity(dist)
            candidates.append((doc, sim, float(dist)))

        debug["candidates_top10"] = [
            {
                "source": c[0].metadata.get("source"),
                "page": c[0].metadata.get("page"),
                "similarity": c[1],
                "distance": c[2],
                "preview": c[0].page_content[:220].replace("\n", " "),
            }
            for c in candidates[:10]
        ]

        filtered_docs = [d for (d, sim, _) in candidates if sim >= MIN_SIMILARITY]
        if not filtered_docs:
            filtered_docs = [d for (d, _, _) in candidates[:max(TOP_K_RESULTS, 4)]]
            debug["threshold_fallback"] = True

        if _reranker is not None and len(filtered_docs) > 1:
            ranked = _reranker.rerank(query, filtered_docs, top_n=min(RERANK_TOP_N, len(filtered_docs)))
            debug["reranker"] = {"enabled": True, "model": RERANKER_MODEL}
            debug["reranked_top"] = [
                {
                    "score": float(score),
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "preview": doc.page_content[:220].replace("\n", " "),
                }
                for doc, score in ranked
            ]
            final_docs = [doc for doc, _ in ranked][:TOP_K_RESULTS]
        else:
            debug["reranker"] = {"enabled": False}
            final_docs = filtered_docs[:TOP_K_RESULTS]

        debug["final_count"] = len(final_docs)
        return final_docs, debug


vector_db = VectorDB()


# ---------------- QA FLOW ---------------- #
def decide_retrieval(query: str, llm) -> str:
    prompt = ChatPromptTemplate.from_template(ROUTING_PROMPT)
    chain = prompt | llm
    out = chain.invoke({"question": query})
    text = out.content.strip().upper() if hasattr(out, "content") else str(out).strip().upper()
    if "DIRECT" in text:
        return "DIRECT"
    return "RETRIEVE"


def answer_direct(query: str, llm, history: List[Dict[str, str]]) -> str:
    prompt = ChatPromptTemplate.from_template(PROMPT_WITHOUT_DOCS)
    chain = prompt | llm
    out = chain.invoke({"question": query, "chat_history": format_chat_history(history)})
    return out.content if hasattr(out, "content") else str(out)


def answer_with_docs(query: str, llm, history: List[Dict[str, str]], docs) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (answer, meta) where meta proves context was passed.
    """
    context, used_chunks = docs_to_context(docs)

    if docs and not context.strip():
        d0 = docs[0]
        src = os.path.basename(str(d0.metadata.get("source", "unknown")))
        context = f"[source={src}]\n{d0.page_content[:2000]}"

    prompt = ChatPromptTemplate.from_template(PROMPT_WITH_DOCS)
    chain = prompt | llm

    out = chain.invoke(
        {
            "question": query,
            "chat_history": format_chat_history(history),
            "context": context,
        }
    )
    answer = out.content if hasattr(out, "content") else str(out)

    meta = {
        "context_chars": len(context),
        "chunks_used": used_chunks,
        "max_context_chars": MAX_CONTEXT_CHARS,
    }
    return answer, meta
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import asyncio

from config import PDFS_DIRECTORY, MAX_FILE_SIZE_BYTES
from rag import (
    AVAILABLE_MODELS,
    make_llm,
    vector_db,
    decide_retrieval,
    answer_direct,
    answer_with_docs,
)

app = FastAPI(title="ResearchGPT Backend", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model_name: str
    history: list = []
    force_retrieval: bool | None = None

@app.get("/api/models")
def get_models():
    return {"models": list(AVAILABLE_MODELS.keys())}

@app.get("/api/status")
def get_status():
    ready = vector_db.load_if_exists()
    return {
        "db_ready": ready,
        "pdf_count": vector_db.pdf_count(),
        "pdfs": vector_db.list_pdfs(),
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    data = await file.read()
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds size limit.")

    os.makedirs(PDFS_DIRECTORY, exist_ok=True)
    save_path = os.path.join(PDFS_DIRECTORY, file.filename)
    with open(save_path, "wb") as f:
        f.write(data)

    return {"ok": True, "filename": file.filename}

@app.post("/api/vectorize")
def vectorize():
    result = vector_db.rebuild()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("reason", "Vectorization failed"))
    return result

@app.get("/api/debug/retrieval")
def debug_retrieval(query: str = Query(..., min_length=1)):
    if not vector_db.load_if_exists():
        raise HTTPException(status_code=400, detail="DB not ready. Vectorize first.")
    _, dbg = vector_db.retrieve(query)
    return dbg

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\n" + f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/api/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    if req.model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Unknown model_name")

    llm = make_llm(req.model_name)

    if req.force_retrieval is None:
        decision = decide_retrieval(req.message, llm)
    else:
        decision = "RETRIEVE" if req.force_retrieval else "DIRECT"

    docs = []
    dbg = None
    rag_meta = None
    retrieved = 0

    if decision == "RETRIEVE":
        if vector_db.load_if_exists():
            docs, dbg = vector_db.retrieve(req.message)
            retrieved = len(docs)
            if docs:
                pass
            else:
                decision = "DIRECT"
        else:
            decision = "DIRECT"

    async def gen():
        yield sse_event("meta", {"decision": decision, "retrieved": retrieved})

        try:
            if decision == "RETRIEVE" and docs:
                try:
                    from langchain_core.prompts import ChatPromptTemplate
                    from rag import PROMPT_WITH_DOCS, docs_to_context, format_chat_history, MAX_CONTEXT_CHARS

                    context, used_chunks = docs_to_context(docs)
                    if docs and not context.strip():
                        d0 = docs[0]
                        src = os.path.basename(str(d0.metadata.get("source", "unknown")))
                        context = f"[source={src}]\n{d0.page_content[:2000]}"

                    rag_meta_local = {
                        "context_chars": len(context),
                        "chunks_used": used_chunks,
                        "max_context_chars": MAX_CONTEXT_CHARS,
                    }
                    yield sse_event("rag_meta", rag_meta_local)
                    if dbg is not None:
                        yield sse_event("debug", dbg)

                    prompt = ChatPromptTemplate.from_template(PROMPT_WITH_DOCS)
                    chain = prompt | llm
                    inputs = {
                        "question": req.message,
                        "chat_history": format_chat_history(req.history),
                        "context": context,
                    }

                    async for chunk in chain.astream(inputs):
                        txt = getattr(chunk, "content", None)
                        if txt is None:
                            txt = str(chunk)
                        if txt:
                            yield sse_event("token", {"text": txt})

                except Exception:
                    answer, rag_meta_local = answer_with_docs(req.message, llm, req.history, docs)
                    yield sse_event("rag_meta", rag_meta_local)
                    if dbg is not None:
                        yield sse_event("debug", dbg)
                    yield sse_event("token", {"text": answer})

            else:
                try:
                    from langchain_core.prompts import ChatPromptTemplate
                    from rag import PROMPT_WITHOUT_DOCS, format_chat_history

                    prompt = ChatPromptTemplate.from_template(PROMPT_WITHOUT_DOCS)
                    chain = prompt | llm
                    inputs = {"question": req.message, "chat_history": format_chat_history(req.history)}

                    async for chunk in chain.astream(inputs):
                        txt = getattr(chunk, "content", None)
                        if txt is None:
                            txt = str(chunk)
                        if txt:
                            yield sse_event("token", {"text": txt})
                except Exception:
                    answer = answer_direct(req.message, llm, req.history)
                    yield sse_event("token", {"text": answer})

            yield sse_event("done", {"ok": True})

        except Exception as e:
            yield sse_event("error", {"message": str(e)})

        if await request.is_disconnected():
            return

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Unknown model_name")

    llm = make_llm(req.model_name)

    if req.force_retrieval is None:
        decision = decide_retrieval(req.message, llm)
    else:
        decision = "RETRIEVE" if req.force_retrieval else "DIRECT"

    if decision == "RETRIEVE":
        if not vector_db.load_if_exists():
            response = answer_direct(req.message, llm, req.history)
            return {"decision": "DIRECT", "response": response, "retrieved": 0, "rag_meta": {"reason": "DB not ready"}}

        docs, dbg = vector_db.retrieve(req.message)
        if not docs:
            response = answer_direct(req.message, llm, req.history)
            return {"decision": "DIRECT", "response": response, "retrieved": 0, "debug": dbg, "rag_meta": {"reason": "no docs"}}

        response, meta = answer_with_docs(req.message, llm, req.history, docs)
        return {"decision": "RETRIEVE", "response": response, "retrieved": len(docs), "debug": dbg, "rag_meta": meta}

    response = answer_direct(req.message, llm, req.history)
    return {"decision": "DIRECT", "response": response, "retrieved": 0}
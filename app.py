# app.py
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi import Body, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from document_understanding import (
    RagServiceConfig,
    chat_stream_no_context,
    process_and_run_stream,
    chat_stream_for_collection,
)

import re

LAST_DOC_COLLECTION: str | None = None

app = FastAPI(title="Mitas Chatbot Demo", version="1.0")

SERVICE_CONFIG = RagServiceConfig.from_env()

def _form_to_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "on", "yes"}

# farklı origin testleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static altında HTML servisi
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")

@app.post("/api/run")
async def api_run(
    file: UploadFile,
    question: str = Form(""),
    preprocess: str = Form("1"),
):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Lütfen bir dosya seçiniz.")

    suffix = Path(file.filename).suffix or ".pdf"
    try:
        contents = await file.read()
    except Exception as exc:  # pragma: no cover - FastAPI IO hatası
        raise HTTPException(status_code=400, detail=f"Dosya okunamadı: {exc}") from exc

    if not contents:
        raise HTTPException(status_code=400, detail="Boş dosya gönderildi.")

    preprocess_flag = _form_to_bool(preprocess)
    config = SERVICE_CONFIG
    if preprocess_flag != SERVICE_CONFIG.preprocess:
        config = replace(SERVICE_CONFIG, preprocess=preprocess_flag)

    upload_dir = Path(tempfile.mkdtemp(prefix="hr_upload_"))
    original_name = Path(file.filename).name or f"yuklenen-belge{suffix}"
    safe_name = original_name.replace("\\", "_").replace("/", "_")
    tmp_path = upload_dir / safe_name
    tmp_path.write_bytes(contents)

    def generate():
        global LAST_DOC_COLLECTION
        try:
            buffer = ""
            col_re = re.compile(r"\[COLLECTION\s+([^\]\s]+)\]")
            for chunk in process_and_run_stream(
                str(tmp_path),
                question=question,
                config=config,
            ):
                # Sunucu tarafında koleksiyon işaretini yakala
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    m = col_re.search(line)
                    if m:
                        LAST_DOC_COLLECTION = m.group(1)
                    yield line + "\n"
            if buffer:
                m = col_re.search(buffer)
                if m:
                    LAST_DOC_COLLECTION = m.group(1)
                yield buffer
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    os.remove(tmp_path)
            shutil.rmtree(upload_dir, ignore_errors=True)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

@app.post("/api/chat")
async def api_chat(prompt: str = Body("", embed=True)):
    question = prompt.strip() or "Selam!"

    def generate():
        for chunk in chat_stream_no_context(question, config=SERVICE_CONFIG):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/api/doc_chat")
async def api_doc_chat(prompt: str = Body("", embed=True)):
    question = prompt.strip()
    if not question:
        question = "Selam!"
    if not LAST_DOC_COLLECTION:
        raise HTTPException(status_code=400, detail="Önce bir belge yükleyip indeksleyiniz (koleksiyon bulunamadı).")

    collection = LAST_DOC_COLLECTION

    def generate():
        yield f"[COLLECTION {collection}]\n"
        for chunk in chat_stream_for_collection(question, collection, config=SERVICE_CONFIG):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

# Sağlık kontrolü
@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

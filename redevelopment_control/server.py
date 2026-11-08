"""HTTP-обёртка для фронта. Запуск: uvicorn redevelopment_control.server:app --port 8000
POST /changes?mode=cnn|verify|vlm  (files: before, after) -> JSON;  POST /entities (file: plan) -> JSON;  GET /health"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import detect_changes, detect_changes_vlm, parse_entities, verify_changes

app = FastAPI(title="redevelopment-control")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # демо: открыто для любого фронта


@app.get("/health")
def health(): return {"ok": True}


@app.post("/changes")
async def changes(before: UploadFile = File(...), after: UploadFile = File(...), mode: str = "cnn", model: str = "claude-sonnet-5-5"):
    """mode: cnn (по умолчанию, локально) | verify (CNN + проверка Claude) | vlm (только Claude). Режимы verify и vlm требуют ключ ANTHROPIC_API_KEY."""
    b, a = await before.read(), await after.read()
    try:
        if mode == "vlm": return detect_changes_vlm(b, a, model)
        res = detect_changes(b, a)
        return verify_changes(b, a, res, model) if mode == "verify" else res
    except ValueError as e: raise HTTPException(400, str(e))
    except Exception as e:  # ошибки API Claude (нет ключа, лимиты, отказ модели) отдаём как 502 с текстом
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.post("/entities")
async def entities(plan: UploadFile = File(...)):
    try: return parse_entities(await plan.read())
    except ValueError as e: raise HTTPException(400, str(e))

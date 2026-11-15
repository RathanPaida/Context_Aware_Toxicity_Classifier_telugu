"""API for the hierarchical model; no domain selector or lexical overrides."""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from contextual.config import load_config, location
from contextual.inference import Predictor

cfg = load_config()
predictor = None
startup_error = None


@asynccontextmanager
async def lifespan(app):
    global predictor, startup_error
    try:
        predictor = Predictor(location(cfg, "checkpoint"))
    except (OSError, ValueError, RuntimeError) as exc:
        startup_error = str(exc)
    yield


app = FastAPI(title="Context-Aware Toxicity", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=cfg["server"]["cors_origins"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class ModerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    history: list[str] = Field(default_factory=list)
    include_embeddings: bool = False


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    parent_id: Optional[str] = None
    conversation_id: str
    text: str = Field(min_length=1)


class ThreadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message]
    target_id: str
    include_embeddings: bool = False


def require_model():
    if predictor is None:
        raise HTTPException(status_code=503, detail=startup_error or "Model is starting")
    return predictor


@app.get("/api/health")
def health():
    return {"ready": predictor is not None, "detail": startup_error,
            "model": cfg["encoder"]["name"], "topic_input": False}


@app.post("/api/moderate")
def moderate(req: ModerateRequest):
    model = require_model()
    try:
        return model.predict(req.text, req.history, req.include_embeddings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/thread")
def thread(req: ThreadRequest):
    model = require_model()
    try:
        return model.predict_thread([m.model_dump() for m in req.messages], req.target_id, req.include_embeddings)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/benchmarks")
def benchmarks():
    path = location(cfg, "report")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No measured evaluation yet")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "contextual.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])

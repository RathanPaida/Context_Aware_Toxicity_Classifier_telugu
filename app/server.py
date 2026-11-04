"""
FastAPI server for the Context-Aware Indic & Telugu Toxicity Classifier.

Run from the repository root:   python app/server.py
Environment:
    PORT                 (default 7860)
    CORS_ORIGINS         comma-separated list (default: http://localhost:3000,http://127.0.0.1:3000)
    ALLOW_UNTRAINED=1    serve an untrained model (UI testing only)
"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import CHECKPOINT_DIR, EVAL_RESULTS_PATH, SCENARIOS_PATH, TOXICITY_LABELS, get_logger
from src.inference_pipeline import ModelNotReadyError, TwoStageModerationPipeline
from src.model import read_checkpoint_metadata

logger = get_logger("web_server")
pipeline: Optional[TwoStageModerationPipeline] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipeline
    pipeline = TwoStageModerationPipeline()
    yield


app = FastAPI(
    title="Indic Shield AI - Context-Aware Toxicity Classifier",
    description="Two-stage moderation cascade for Indian & Telugu code-mixed text",
    version="2.1.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


class ModerateRequest(BaseModel):
    text: str
    thread_id: str = "default_thread"
    topic: Optional[str] = None
    parent_context: Optional[str] = None   # explicit parent for single-comment deep dive
    history: Optional[List[str]] = None    # explicit multi-turn context (oldest → newest)
    update_history: bool = False


class ClearThreadRequest(BaseModel):
    thread_id: str


def _require_pipeline() -> TwoStageModerationPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is starting up")
    return pipeline


@app.get("/api/health")
def health_check():
    p = pipeline
    info = p.info() if p else {"ready": False, "status": "starting"}
    info["detail"] = info.pop("status")
    return {
        **info,
        "status": "ready" if info["ready"] else "not_ready",
        "labels": TOXICITY_LABELS,
        "timestamp": time.time(),
    }


@app.post("/api/moderate")
def moderate_comment(req: ModerateRequest):
    p = _require_pipeline()
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty comment text")

    override = None
    if req.history:
        override = [h for h in req.history if h and h.strip()]
    elif req.parent_context and req.parent_context.strip():
        override = [req.parent_context.strip()]

    try:
        result = p.moderate(
            text=text,
            thread_id=req.thread_id,
            topic=req.topic,
            update_history=req.update_history and override is None,
            history_override=override,
        )
    except ModelNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        **result,
        "raw_text": text,
        "cleaned_text": p.preprocessor.clean_text(text),
        "thread_id": req.thread_id,
        "topic": req.topic,
        "history": override if override is not None else p.cache.get_history(req.thread_id, p.cfg.history_turns),
    }


@app.post("/api/clear-thread")
def clear_thread(req: ClearThreadRequest):
    _require_pipeline().clear_thread(req.thread_id)
    return {"status": "success", "thread_id": req.thread_id}


SLICE_INFO = {
    "contrastive_flip": ("Contrastive Context Flip", "Same target under safe vs toxic parent context", "Exact twin match"),
    "identity_bias": ("Caste & Regional Identity Safety", "Neutral mentions of caste / regional communities", "Subgroup false-positive rate"),
    "parent_bleed": ("Parent-Bleed Robustness", "Polite de-escalation after a toxic parent", "De-escalation specificity"),
    "script_parity": ("Script Parity", "Romanized vs native Telugu script", "|ΔF1|"),
    "latency_p50": ("Latency", "Median model-path latency on the evaluation machine", "p50 latency"),
}


@app.get("/api/benchmarks")
def get_benchmarks():
    """Measured results from `python run_pipeline.py --evaluate` (reports/eval_results.json)."""
    stages = []
    for i, name in enumerate(["stage1_best", "stage2_best", "stage3_best", "stage4_best"], start=1):
        meta = read_checkpoint_metadata(CHECKPOINT_DIR / name)
        if not meta:
            stages.append({"stage": i, "name": name, "data": "-", "metric": "not trained", "checkpoint": name})
            continue
        vm = meta.get("val_metrics", {})
        if "macro_f1@threshold" in vm:
            metric = f"val macro-F1 {vm['macro_f1@threshold']:.3f}"
            if "flip_accuracy" in vm:
                metric += f" | held-out flip {vm['flip_accuracy']:.0%}"
        elif "perplexity" in meta:
            metric = f"val PPL {meta['perplexity']:.1f}"
        else:
            metric = "trained"
        stages.append({"stage": i, "name": name, "data": meta.get("data", "-"), "metric": metric,
                       "checkpoint": name, "trained_at": meta.get("timestamp")})

    if not EVAL_RESULTS_PATH.exists():
        return {"evaluated": False, "message": "No evaluation results yet — run `python run_pipeline.py --evaluate`.",
                "slices": [], "training_stages": stages}

    res = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    slices = []
    for key, v in res.get("slices", {}).items():
        name, desc, metric = SLICE_INFO.get(key, (key, "", key))
        slices.append({
            "id": key, "name": name, "description": f"{desc} (n={v['n']})", "metric": metric,
            "value": f"{v['value']}{v['unit']}", "gate": v["gate"],
            "status": "PASSED" if v["passed"] else "FAILED", "passed": v["passed"],
        })
    ho = res.get("heldout_test")
    if ho:
        slices.insert(0, {
            "id": "heldout_test", "name": "Held-out Test Set (real data)",
            "description": f"DravidianCodeMix + Cultural-Safety-Telugu test split (n={ho['n']:,})",
            "metric": "Macro-F1 (tuned thresholds)", "value": f"{ho['macro_f1@threshold']:.3f}",
            "gate": "-", "status": "MEASURED", "passed": True,
        })
    return {"evaluated": True, "evaluated_at": res.get("evaluated_at"), "slices": slices, "training_stages": stages}


@app.get("/api/scenarios")
def get_preset_scenarios():
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "Indic Shield AI API is running."})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting Indic Shield AI on http://localhost:{port}")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port, reload=False)

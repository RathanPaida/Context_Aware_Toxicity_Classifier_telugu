"""All filesystem locations are resolved against this repository, never cwd."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "contextual.json"


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path=None):
    path = project_path(path or os.environ.get("TOXICITY_CONFIG", DEFAULT_CONFIG))
    cfg = json.loads(path.read_text(encoding="utf-8"))
    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    for key in ("hidden_size", "heads", "layers", "feedforward_size", "max_turns"):
        if not isinstance(m[key], int) or m[key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    if m["hidden_size"] % m["heads"]:
        raise ValueError("hidden_size must be divisible by heads")
    if not 0 <= m["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    for key in ("epochs", "batch_size", "patience", "cpu_threads"):
        if t[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if t["learning_rate"] <= 0 or t["gradient_clip"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if cfg["encoder"]["max_tokens"] <= 0 or cfg["encoder"]["batch_size"] <= 0:
        raise ValueError("Invalid encoder settings")
    if not (0 < d["validation_fraction"] < 1 and 0 < d["test_fraction"] < 1 and
            d["validation_fraction"] + d["test_fraction"] < 1):
        raise ValueError("Invalid split fractions")
    return cfg


def location(cfg, key):
    return project_path(cfg["paths"][key])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

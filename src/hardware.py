"""
Hardware detection: pick training settings that fit the machine, and check the environment
before a long unattended run.

    python run_pipeline.py --check-env

Profiles keep the *effective* batch size at 32 (batch × gradient accumulation) so results
are comparable across GPUs; only memory use and speed change.
"""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from src.config import MIN_FREE_DISK_GB, PROJECT_ROOT, get_logger

logger = get_logger("hardware")

EFFECTIVE_BATCH = 32
MLM_EFFECTIVE_BATCH = 32
REQUIRED_PACKAGES = ["torch", "transformers", "datasets", "pandas", "pyarrow", "sklearn", "fastapi", "uvicorn", "tqdm"]


@dataclass
class HardwareProfile:
    tier: str                     # cpu | small | medium | large
    device: str
    gpu_name: Optional[str]
    vram_gb: float
    free_disk_gb: float
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    freeze_embeddings: bool
    max_seq_length: int
    mlm_batch_size: int
    mlm_gradient_accumulation_steps: int
    mlm_freeze_embeddings: bool

    def describe(self) -> str:
        gpu = f"{self.gpu_name} ({self.vram_gb:.1f} GB)" if self.gpu_name else "no CUDA GPU"
        return (f"{gpu} → tier '{self.tier}': batch {self.batch_size}×{self.gradient_accumulation_steps} "
                f"(effective {self.batch_size * self.gradient_accumulation_steps}), grad-ckpt "
                f"{'on' if self.gradient_checkpointing else 'off'}, embeddings "
                f"{'frozen' if self.freeze_embeddings else 'trained'}, seq {self.max_seq_length}; "
                f"MLM batch {self.mlm_batch_size}×{self.mlm_gradient_accumulation_steps} | "
                f"free disk {self.free_disk_gb:.0f} GB")


def free_disk_gb(path=PROJECT_ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def profile_for(vram_gb: Optional[float], gpu_name: Optional[str] = None, disk_gb: float = 0.0) -> HardwareProfile:
    """Pure function (testable): settings for a given amount of VRAM (None = no GPU)."""
    if vram_gb is None:
        tier, batch, ckpt, freeze, mlm_batch, mlm_freeze = "cpu", 8, False, True, 8, True
    elif vram_gb < 7.5:      # 4–6 GB laptop GPUs (RTX 3050/4050/2060 …)
        tier, batch, ckpt, freeze, mlm_batch, mlm_freeze = "small", 8, True, True, 8, True
    elif vram_gb < 14:       # 8–12 GB (RTX 3060/4060/4070 …)
        tier, batch, ckpt, freeze, mlm_batch, mlm_freeze = "medium", 16, False, True, 16, False
    else:                    # 16 GB+ (RTX 4080/4090, A100, T4×2 …)
        tier, batch, ckpt, freeze, mlm_batch, mlm_freeze = "large", 32, False, False, 32, False
    return HardwareProfile(
        tier=tier, device="cpu" if vram_gb is None else "cuda", gpu_name=gpu_name, vram_gb=vram_gb or 0.0,
        free_disk_gb=disk_gb, batch_size=batch, gradient_accumulation_steps=max(1, EFFECTIVE_BATCH // batch),
        gradient_checkpointing=ckpt, freeze_embeddings=freeze, max_seq_length=256,
        mlm_batch_size=mlm_batch, mlm_gradient_accumulation_steps=max(1, MLM_EFFECTIVE_BATCH // mlm_batch),
        mlm_freeze_embeddings=mlm_freeze,
    )


def detect_profile() -> HardwareProfile:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return profile_for(props.total_memory / 1e9, props.name, free_disk_gb())
    return profile_for(None, None, free_disk_gb())


def gpu_processes() -> List[Tuple[str, str]]:
    """(pid, name) of processes using the GPU, via nvidia-smi. Empty if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2 and parts[0] != str(os.getpid()):
            procs.append((parts[0], parts[1]))
    return procs


def _hf_reachable() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/ai4bharat/IndicBERTv2-MLM-Sam-TLM", timeout=8)
        return True
    except Exception:
        return False


def check_environment(allow_cpu: bool = False) -> bool:
    """Print an environment report. Returns False if a blocking problem was found."""
    import torch
    ok = True
    prof = detect_profile()
    lines = [
        f"Python {platform.python_version()} on {platform.system()} {platform.release()}",
        f"torch {torch.__version__} | CUDA build {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}",
        f"Profile: {prof.describe()}",
    ]
    problems, warnings = [], []

    missing = [p for p in REQUIRED_PACKAGES if importlib.util.find_spec(p) is None]
    if missing:
        problems.append(f"missing packages: {missing} → pip install -r requirements.txt")
    if importlib.util.find_spec("indic_transliteration") is None:
        warnings.append("indic-transliteration not installed → no Romanized-Telugu augmentation (pip install -r requirements.txt)")
    if importlib.util.find_spec("indicnlp") is None:
        warnings.append("indic-nlp-library not installed → only basic Unicode normalisation")

    if prof.device == "cpu" and not allow_cpu:
        problems.append("no CUDA GPU detected — training would take days. Install a CUDA build of torch "
                        "(https://pytorch.org/get-started/locally/) or pass --allow-cpu")
    if prof.free_disk_gb < MIN_FREE_DISK_GB:
        problems.append(f"only {prof.free_disk_gb:.1f} GB free disk; need ≥ {MIN_FREE_DISK_GB} GB "
                        "(model download ~1.1 GB, datasets ~1 GB, checkpoints ~0.6 GB each + resume state ~1.2 GB)")
    others = gpu_processes()
    if others:
        warnings.append("other processes are using the GPU (close them to free VRAM — browsers with hardware "
                        "acceleration, Jupyter kernels, old training runs): "
                        + ", ".join(f"{pid}:{os.path.basename(name)}" for pid, name in others))
    if not _hf_reachable():
        problems.append("cannot reach huggingface.co (needed to download the model and datasets)")

    logger.info("=" * 70)
    logger.info("ENVIRONMENT CHECK")
    for l in lines:
        logger.info("  " + l)
    for w in warnings:
        logger.warning("  ⚠ " + w)
    for p in problems:
        logger.error("  ✗ " + p)
        ok = False
    logger.info("  ✓ environment OK" if ok else "  ✗ fix the problems above before training")
    logger.info("=" * 70)
    return ok


def profile_dict() -> dict:
    return asdict(detect_profile())


if __name__ == "__main__":
    sys.exit(0 if check_environment() else 1)

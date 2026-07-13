#!/usr/bin/env python3
"""Create the local UniverSR ONNX model used by Sidespread."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "external" / "universr"
VENV = ROOT / ".venv"
UNIVERSR_REVISION = "26dc21c44e11f9f19e823f02b0d4641dd5ea5af2"
EXPORT_DEPENDENCIES = [
    "einops>=0.7",
    "huggingface_hub>=0.20",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "pyyaml>=6.0",
    "timm>=0.9",
    "torch>=2.0",
    "torchaudio>=2.0",
    "torchdiffeq>=0.2.3",
    "tqdm>=4.60",
]


def run(*args: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> None:
    if not (UPSTREAM / ".git").is_dir():
        UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/woongzip1/UniverSR.git", str(UPSTREAM))

    run("git", "fetch", "origin", UNIVERSR_REVISION, "--depth=1", cwd=UPSTREAM)
    run("git", "checkout", "--detach", UNIVERSR_REVISION, cwd=UPSTREAM)

    python = venv_python()
    if not python.exists():
        run(sys.executable, "-m", "venv", str(VENV))

    run(str(python), "-m", "pip", "install", "--upgrade", "pip")
    run(str(python), "-m", "pip", "install", *EXPORT_DEPENDENCIES)
    run(str(python), str(ROOT / "scripts" / "export_universr_onnx.py"))

    model = ROOT / "models" / "universr_backbone.onnx"
    print(f"\nModel ready: {model}")


if __name__ == "__main__":
    main()

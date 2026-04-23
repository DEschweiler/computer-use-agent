#!/usr/bin/env python3
"""
One-time model setup for computer-use-agent.

1. RapidOCR  — instantiates the engine so it downloads and caches its models
               (~30-50 MB) to ~/.rapidocr/models/.
2. OmniParser YOLO — downloads icon_detect/model.pt from HuggingFace Hub,
               exports to ONNX, and saves to models/icon_detect.onnx.

Run once per environment before starting the agent:
    conda activate visualagent
    python download_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
ONNX_PATH = MODELS_DIR / "icon_detect.onnx"
YOLO_IMGSZ = 1280  # default imgsz; dynamic export supports any size at runtime


def setup_rapidocr() -> None:
    print("=" * 60)
    print("RapidOCR models")
    print("=" * 60)
    try:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    except ImportError:
        sys.exit("rapidocr not installed. Run: pip install rapidocr onnxruntime")

    print("Initializing RapidOCR engine (downloads models on first run) ...")
    t0 = time.perf_counter()
    engine = RapidOCR(params={
        "Rec.lang_type": LangRec.LATIN,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_type": ModelType.MOBILE,
        "Global.log_level": "error",
    })
    elapsed = time.perf_counter() - t0

    # Run a tiny dummy inference to confirm the models work end-to-end.
    import numpy as np
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    engine(dummy)
    print(f"[OK] RapidOCR ready in {elapsed:.1f}s  (models cached in ~/.rapidocr/models/)\n")


def setup_yolo() -> None:
    print("=" * 60)
    print("OmniParser YOLO icon detector")
    print("=" * 60)
    MODELS_DIR.mkdir(exist_ok=True)

    if ONNX_PATH.exists():
        print(f"[OK] ONNX already present: {ONNX_PATH}")
        print("     Delete it and re-run this script to force a fresh export.")
        _verify_yolo()
        return

    # 1. Download .pt from HuggingFace Hub
    print("Downloading icon_detect/model.pt from microsoft/OmniParser-v2.0 ...")
    print("(~100 MB — cached in your HF cache dir after the first download)\n")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    t0 = time.perf_counter()
    pt_path = hf_hub_download(
        repo_id="microsoft/OmniParser-v2.0",
        filename="icon_detect/model.pt",
    )
    print(f"Downloaded in {time.perf_counter() - t0:.1f}s: {pt_path}\n")

    # 2. Export to ONNX
    print(f"Exporting to ONNX (imgsz={YOLO_IMGSZ}) — takes ~30 s on first run ...")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    t0 = time.perf_counter()
    exported = YOLO(pt_path).export(
        format="onnx",
        imgsz=YOLO_IMGSZ,
        simplify=True,
        dynamic=True,
    )
    tmp_onnx = Path(exported)
    print(f"Export complete in {time.perf_counter() - t0:.1f}s: {tmp_onnx}\n")

    tmp_onnx.rename(ONNX_PATH)
    print(f"Saved to: {ONNX_PATH}\n")

    _verify_yolo()


def _verify_yolo() -> None:
    print("Verifying ONNX with onnxruntime ...")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping verification.")
        return

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(ONNX_PATH), sess_options=so, providers=["CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape

    import numpy as np
    dummy = np.zeros((1, 3, YOLO_IMGSZ, YOLO_IMGSZ), dtype=np.float32)
    t0 = time.perf_counter()
    out = sess.run(None, {inp_name: dummy})
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Input : {inp_name!r}  shape={inp_shape}")
    print(f"  Output: {[o.shape for o in out]}")
    print(f"  First inference: {elapsed:.1f} ms  (cold, includes graph-opt warmup)")
    print(f"[OK] YOLO model ready at models/icon_detect.onnx\n")


def main() -> None:
    setup_rapidocr()
    setup_yolo()
    print("All models ready. You can now start the agent.")


if __name__ == "__main__":
    main()

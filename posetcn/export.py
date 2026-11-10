"""Export a trained run for the mobile app.

    python -m posetcn.export runs/<name>

Writes into runs/<name>/export/:
    posture_tcn.onnx       input "features" (1, window, F) -> output "logits" (1, C); apply SIGMOID per output
    posture_tcn.tflite     if `ai-edge-torch` is installed (pip install ai-edge-torch)
    posture_tcn.pt         TorchScript (for a laptop/server fallback)
    model_spec.json        everything the app needs to reproduce preprocessing + alerts
Feature normalisation is baked into the model; the app only ports features.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .features import FEATURE_NAMES, LEFT_IDX, NUM_FEATURES, RIGHT_IDX
from .model import load_run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)

    model, cfg, ckpt = load_run(args.run_dir, "cpu")
    W = cfg["data"]["window"]
    out = args.run_dir / "export"
    out.mkdir(exist_ok=True)
    x = torch.randn(4, W, NUM_FEATURES)
    with torch.no_grad():
        ref = model(x).numpy()

    ts = torch.jit.trace(model, x[:1])
    ts.save(str(out / "posture_tcn.pt"))
    print("wrote posture_tcn.pt")

    onnx_path = out / "posture_tcn.onnx"
    try:
        torch.onnx.export(model, (x[:1],), str(onnx_path), input_names=["features"], output_names=["logits"],
                          opset_version=17, dynamo=False)
        print("wrote", onnx_path.name)
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path))
            got = np.concatenate([sess.run(None, {"features": x[i:i + 1].numpy()})[0] for i in range(len(x))])
            print(f"  onnx parity max|diff| = {np.abs(got - ref).max():.2e}")
        except ImportError:
            print("  (pip install onnxruntime to verify ONNX parity)")
    except Exception as e:  # noqa: BLE001
        print(f"ONNX export failed ({e}); pip install onnx")

    try:
        import ai_edge_torch
        edge = ai_edge_torch.convert(model, (x[:1],))
        edge.export(str(out / "posture_tcn.tflite"))
        got = np.concatenate([edge(x[i:i + 1]) for i in range(len(x))])
        print(f"wrote posture_tcn.tflite  (parity max|diff| = {np.abs(got - ref).max():.2e})")
    except ImportError:
        print("TFLite skipped: pip install ai-edge-torch (Linux/macOS), or convert the ONNX with onnx2tf")

    spec = {
        "outputs": cfg["classes"],
        "output_activation": "sigmoid (multi-label; good posture = all faults below threshold)",
        "good_tag": cfg.get("good_tag"),
        "gate_class": cfg.get("gate_class"),
        "alert_zones": cfg["alert_zones"],
        "alerts": cfg["alerts"],
        "fps": cfg["data"]["fps"],
        "window": W,
        "num_features": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "mediapipe_indices": {"left": LEFT_IDX, "right": RIGHT_IDX,
                              "order": ["nose", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"]},
        "preprocessing": (
            "Port posetcn/features.py: choose side by visibility over the first ~2 s, canonicalise to pixels "
            "with y up and facing +x, ref_scale = running median torso length (~60 s), compute features on the "
            "last `window` frames (delta of first frame = 0). Normalisation is inside the model."),
        "val_metrics": {"mean_fault_f1": ckpt["val"]["mean_fault_f1"], "good_f1": ckpt["val"]["good"]["f1"]},
    }
    with open(out / "model_spec.json", "w") as f:
        json.dump(spec, f, indent=2)
    print("wrote model_spec.json")


if __name__ == "__main__":
    main()

# gradio_app.py
import os, io, json, time
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
import joblib

import gradio as gr

# --- Config ---
MODEL_DIR = Path("model")
FS_DIR = Path("fs_saved")
IMG_SIZE = (224, 224)
TOP_K = 5

# Lazy-loaded objects
_dl_model = None
_idx2label = None
_classical_clf = None
_selector = None
_extractor = None

# --- Helpers ---
def load_class_indices():
    global _idx2label
    p = MODEL_DIR / "class_indices.json"
    if p.exists():
        with open(p, "r") as f:
            cls = json.load(f)
        # cls: {label: idx} -> invert to idx->label
        _idx2label = {int(v): k for k, v in cls.items()}
    else:
        _idx2label = None
    return _idx2label

def load_dl_model():
    global _dl_model
    if _dl_model is None:
        # prefer .keras then .h5
        if (MODEL_DIR / "best_model.keras").exists():
            path = MODEL_DIR / "best_model.keras"
        elif (MODEL_DIR / "best_model.h5").exists():
            path = MODEL_DIR / "best_model.h5"
        else:
            raise FileNotFoundError("No DL model found in model/ (best_model.keras or best_model.h5)")
        import tensorflow as tf
        _dl_model = tf.keras.models.load_model(str(path))
    return _dl_model

def ensure_classical_assets():
    global _classical_clf, _selector, _extractor
    if _classical_clf is None:
        # pick the first joblib in fs_saved that looks like a classifier
        candidates = list(FS_DIR.glob("*_variance_fs.joblib")) + list(FS_DIR.glob("best_classical_*.joblib")) + list(FS_DIR.glob("*.joblib"))
        if not candidates:
            raise FileNotFoundError("No classical model found in fs_saved/")
        _classical_clf = joblib.load(str(candidates[0]))
    if _selector is None:
        sel_candidates = list(FS_DIR.glob("variance_selector*.joblib")) + list(FS_DIR.glob("*.joblib"))
        if sel_candidates:
            _selector = joblib.load(str(sel_candidates[0]))
    if _extractor is None:
        # lightweight extractor for embeddings
        from tensorflow.keras.applications import MobileNetV2
        _extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(*IMG_SIZE,3))
    load_class_indices()
    return _classical_clf, _selector, _extractor

def pil_preprocess(img: Image.Image, size=IMG_SIZE) -> np.ndarray:
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def topk_from_probs(probs: np.ndarray, idx2label: dict, k=5):
    k = min(k, len(probs))
    top_idx = np.argsort(probs)[-k:][::-1]
    labels = [idx2label.get(int(i), str(i)) for i in top_idx]
    vals = [float(probs[int(i)]) for i in top_idx]
    return list(zip(labels, vals))

# --- Predict function used by Gradio ---
def predict(image: Image.Image, pipeline: str = "dl", top_k: int = TOP_K) -> Tuple[Image.Image, List[Tuple[str,float]]]:
    """
    Returns: (display image, list of (label, confidence))
    """
    start = time.time()
    if pipeline == "dl":
        model = load_dl_model()
        idx2 = load_class_indices()
        inp = pil_preprocess(image)
        probs = model.predict(inp)[0]
        if idx2 is None:
            idx2 = {i: str(i) for i in range(len(probs))}
        topk = topk_from_probs(probs, idx2, k=top_k)
    else:
        clf, sel, extractor = ensure_classical_assets()
        inp = pil_preprocess(image)
        emb = extractor.predict(inp)
        if sel is not None:
            emb_sel = sel.transform(emb)
        else:
            emb_sel = emb
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(emb_sel)[0]
            idx2 = load_class_indices()
            if idx2 is None:
                # fallback to classes_ if available
                try:
                    labels = list(clf.classes_)
                    idx2 = {i: str(labels[i]) for i in range(len(labels))}
                except Exception:
                    idx2 = {i: str(i) for i in range(len(probs))}
            topk = topk_from_probs(probs, idx2, k=top_k)
        else:
            pred = clf.predict(emb_sel)[0]
            topk = [(str(pred), 1.0)]
    # Return the original image and predictions
    took = time.time() - start
    return image, topk

# --- Gradio UI ---
title = "AgroAI — Leaf disease classifier"
description = "Upload a leaf image. Choose pipeline: Deep (CNN) or Classical (embeddings → classifier)."

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Input leaf image")
            pipeline = gr.Dropdown(choices=["dl","classical"], value="dl", label="Pipeline")
            topk_slider = gr.Slider(minimum=1, maximum=10, step=1, value=TOP_K, label="Top K")
            btn = gr.Button("Predict")
        with gr.Column(scale=1):
            out_img = gr.Image(type="pil", label="Uploaded image")
            out_table = gr.Label(num_top_classes=5, label="Top-K predictions")
            out_html = gr.HTML()

    def on_predict(img, pipe, tk):
        if img is None:
            return None, None, "<p style='color:red'>Please upload an image.</p>"
        image, preds = predict(img, pipeline=pipe, top_k=int(tk))
        # Prepare label dict for Gradio.Label
        label_dict = {lab: conf for lab, conf in preds}
        html = "<h4>Top Predictions</h4><ul>"
        for lab,conf in preds:
            html += f"<li>{lab}: {conf:.4f}</li>"
        html += "</ul>"
        return image, label_dict, html

    btn.click(on_predict, inputs=[img_in, pipeline, topk_slider], outputs=[out_img, out_table, out_html])

# Launch when executed on Spaces; do not call demo.launch() here (Spaces auto-launches)
if __name__ == "__main__":
    demo.launch()

# app.py
import io
import os
import json
import time
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import joblib

# -------- CONFIG --------
MODEL_DIR = Path("model")                 # where you saved best_model.keras / .h5
FS_DIR = Path("fs_saved")                 # where classical models/selectors are saved
IMAGE_DIR = Path("static", "uploads")     # folder to store uploaded images
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)                     # must match what your model expects
TOP_K = 5

# -------- APP --------
app = FastAPI(title="AgroAI - Prediction API")

# allow CORS from anywhere for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static files (images) and templates
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------- Load models (lazy load) --------
_dl_model = None
_classical_model = None
_selector = None
_extractor = None
_class_indices = None   # map label->index saved as JSON earlier
_idx2label = None

def load_dl_model():
    global _dl_model
    if _dl_model is None:
        # prefer .keras then .h5
        if (MODEL_DIR / "best_model.keras").exists():
            _dl_model = load_model(str(MODEL_DIR / "best_model.keras"))
        elif (MODEL_DIR / "best_model.h5").exists():
            _dl_model = load_model(str(MODEL_DIR / "best_model.h5"))
        else:
            raise FileNotFoundError("No DL model found in model/ (best_model.keras or .h5)")
    return _dl_model

def load_classical_assets():
    global _classical_model, _selector, _extractor, _class_indices, _idx2label
    if _classical_model is None:
        # classifier (joblib)
        clf_path = FS_DIR / "best_classical_SVM.joblib"
        # try some common names if you saved differently
        if not clf_path.exists():
            # search any joblib in fs_saved
            cands = list(FS_DIR.glob("*_variance_fs.joblib")) + list(FS_DIR.glob("best_classical_*.joblib"))
            if len(cands) == 0:
                raise FileNotFoundError("No classical model found in fs_saved/")
            clf_path = cands[0]
        _classical_model = joblib.load(str(clf_path))

    if _selector is None:
        sel_candidates = list(FS_DIR.glob("variance_selector*.joblib"))
        if len(sel_candidates) == 0:
            sel_candidates = list(FS_DIR.glob("*.joblib"))
        if len(sel_candidates) == 0:
            raise FileNotFoundError("No selector found in fs_saved/")
        # pick the most likely (variance_selector.joblib if present)
        sel_path = [p for p in sel_candidates if "variance_selector" in p.name]
        sel_path = sel_path[0] if sel_path else sel_candidates[0]
        _selector = joblib.load(str(sel_path))

    if _extractor is None:
        # embedding extractor used for training (MobileNetV2 avg pooling in our notebook)
        from tensorflow.keras.applications import MobileNetV2
        _extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(*IMG_SIZE,3))

    if _class_indices is None:
        class_idx_path = FS_DIR.parent.joinpath("model").joinpath("class_indices.json")
        # try other locations
        if not class_idx_path.exists():
            # try top-level model/class_indices.json or saved in current dir
            alt = Path("model/class_indices.json")
            if alt.exists():
                class_idx_path = alt
            else:
                # fallback: try fs_saved/class_indices.json
                alt2 = FS_DIR / "class_indices.json"
                class_idx_path = alt2 if alt2.exists() else None
        if class_idx_path is None or not class_idx_path.exists():
            # attempt to load any json with class_indices
            for j in Path(".").glob("**/class_indices.json"):
                class_idx_path = j
                break
            if class_idx_path is None:
                # as last resort, build numeric labels from 0..N-1
                _class_indices = None
                _idx2label = None
        if class_idx_path and class_idx_path.exists():
            with open(class_idx_path, "r") as f:
                _class_indices = json.load(f)
                _idx2label = {int(v): k for k, v in _class_indices.items()}
    return _classical_model, _selector, _extractor

# -------- Helpers --------
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

def preprocess_for_dl(img: Image.Image):
    # resize and scale
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def preprocess_for_extractor(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def topk_from_probs(probs: np.ndarray, idx2label: dict, k=5):
    top_idx = probs.argsort()[-k:][::-1]
    labels = [idx2label.get(int(i), str(i)) for i in top_idx]
    vals = [float(probs[int(i)]) for i in top_idx]
    return list(zip(labels, vals))

# -------- Routes --------
@app.get("/", response_class=HTMLResponse)
async def index():
    html = Path("templates/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.post("/predict")
async def predict(file: UploadFile = File(...), pipeline: str = Form("dl"), top_k: int = Form(TOP_K)):
    """
    Accepts multipart/form-data with:
      - file: image file
      - pipeline: 'dl' or 'classical' (default 'dl')
      - top_k: integer
    Returns JSON with predictions and a URL to the saved uploaded image.
    """
    start = time.time()
    contents = await file.read()
    img = read_imagefile(contents)

    # save uploaded file for UI
    fname = f"{int(time.time()*1000)}_{file.filename}"
    save_path = IMAGE_DIR / fname
    img.save(save_path)

    try:
        if pipeline == "dl":
            model = load_dl_model()
            inp = preprocess_for_dl(img)
            probs = model.predict(inp)[0]
            # need idx2label mapping: if model was trained with train_gen.class_indices saved
            # try to load class_indices.json from model folder
            ci_path = Path("model/class_indices.json")
            if ci_path.exists():
                with open(ci_path) as f:
                    class_indices = json.load(f)
                    idx2label = {int(v): k for k, v in class_indices.items()}
            else:
                # fallback: assume integer-labeled
                idx2label = {i: str(i) for i in range(len(probs))}
            topk = topk_from_probs(probs, idx2label, k=top_k)
            took = time.time() - start
            return JSONResponse({
                "pipeline": "dl",
                "predictions": [{"label": l, "confidence": p} for l,p in topk],
                "image_url": f"/static/uploads/{fname}",
                "took": took
            })
        elif pipeline == "classical":
            clf, selector, extractor = load_classical_assets()
            inp = preprocess_for_extractor(img)
            emb = extractor.predict(inp)[0:1]  # shape (1,features)
            emb_sel = selector.transform(emb)
            probs = None
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(emb_sel)[0]
                # classifier might need idx2label mapping:
                if _idx2label is not None:
                    idx2label = _idx2label
                else:
                    # try to infer classes from classifier.classes_
                    try:
                        labels = clf.classes_
                        idx2label = {i: str(labels[i]) for i in range(len(labels))}
                    except:
                        idx2label = {i: str(i) for i in range(len(probs))}
                topk = topk_from_probs(probs, idx2label, k=top_k)
            else:
                # classifier only supports predict (no proba). We'll return predicted label and dummy confidence.
                pred = clf.predict(emb_sel)[0]
                label = str(pred)
                topk = [(label, 1.0)]
            took = time.time() - start
            return JSONResponse({
                "pipeline": "classical",
                "predictions": [{"label": l, "confidence": p} for l,p in topk],
                "image_url": f"/static/uploads/{fname}",
                "took": took
            })
        else:
            return JSONResponse({"error": "Unknown pipeline. Use 'dl' or 'classical'."}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Static route to show the uploaded image files (FastAPI StaticFiles already mounted under /static)
@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    fpath = IMAGE_DIR / filename
    if fpath.exists():
        return FileResponse(str(fpath))
    return JSONResponse({"error": "file not found"}, status_code=404)

# Run via: uvicorn app:app --host 0.0.0.0 --port 8000

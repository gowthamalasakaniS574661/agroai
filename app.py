# app.py (robust / defensive)
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

# Defensive imports for optional dependencies
HAS_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
except Exception as e:
    HAS_TF = False
    tf = None
    load_model = None
    # define a placeholder img_to_array using numpy if TF not present
    def img_to_array(img):
        return np.asarray(img)

HAS_JOBLIB = True
try:
    import joblib
except Exception:
    joblib = None
    HAS_JOBLIB = False

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
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -------- Load models (lazy load) --------
_dl_model = None
_classical_model = None
_selector = None
_extractor = None
_class_indices = None   # map label->index saved as JSON earlier
_idx2label = None

def load_dl_model():
    """
    Lazy load the DL model. Raises FileNotFoundError if model not present.
    If TensorFlow is not installed, raises RuntimeError.
    """
    global _dl_model
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in the environment. DL pipeline disabled.")
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
    """
    Lazy load classical classifier, selector, and embedding extractor.
    Requires joblib (and TensorFlow if extractor is a TF model).
    """
    global _classical_model, _selector, _extractor, _class_indices, _idx2label

    if not HAS_JOBLIB:
        raise RuntimeError("joblib not installed; classical pipeline unavailable.")

    # load classifier
    if _classical_model is None:
        clf_path = FS_DIR / "best_classical_SVM.joblib"
        if not clf_path.exists():
            # search common filenames
            cands = list(FS_DIR.glob("*_variance_fs.joblib")) + list(FS_DIR.glob("best_classical_*.joblib")) + list(FS_DIR.glob("*.joblib"))
            if len(cands) == 0:
                raise FileNotFoundError("No classical model found in fs_saved/")
            clf_path = cands[0]
        _classical_model = joblib.load(str(clf_path))

    # load selector
    if _selector is None:
        sel_candidates = list(FS_DIR.glob("variance_selector*.joblib")) + list(FS_DIR.glob("selector*.joblib")) + list(FS_DIR.glob("*.joblib"))
        sel_candidates = [p for p in sel_candidates if p.exists()]
        if len(sel_candidates) == 0:
            raise FileNotFoundError("No selector found in fs_saved/")
        # prefer explicit variance selector if present
        sel_path = next((p for p in sel_candidates if "variance_selector" in p.name), sel_candidates[0])
        _selector = joblib.load(str(sel_path))

    # load extractor (optional). If TF not available, raise informative error.
    if _extractor is None:
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required to construct the embedding extractor used by classical pipeline.")
        from tensorflow.keras.applications import MobileNetV2
        _extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(*IMG_SIZE,3))

    # load class indices mapping if present
    if _class_indices is None:
        # common locations to check
        candidates = [
            FS_DIR.parent.joinpath("model").joinpath("class_indices.json"),
            Path("model/class_indices.json"),
            FS_DIR / "class_indices.json",
            Path("class_indices.json")
        ]
        found = None
        for c in candidates:
            if c and c.exists():
                found = c
                break
        if found:
            try:
                with open(found, "r") as f:
                    _class_indices = json.load(f)
                    _idx2label = {int(v): k for k, v in _class_indices.items()}
            except Exception:
                _class_indices = None
                _idx2label = None
        else:
            _class_indices = None
            _idx2label = None

    return _classical_model, _selector, _extractor

# -------- Helpers --------
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

def preprocess_for_dl(img: Image.Image):
    # resize and scale
    img_r = img.resize(IMG_SIZE)
    arr = img_to_array(img_r) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def preprocess_for_extractor(img: Image.Image):
    img_r = img.resize(IMG_SIZE)
    arr = img_to_array(img_r) / 255.0
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
    # Return templates/index.html if exists, else minimal HTML
    t = Path("templates/index.html")
    if t.exists():
        return HTMLResponse(content=t.read_text(encoding="utf-8"))
    return HTMLResponse(content="<html><body><h3>AgroAI API</h3><p>Use POST /predict with multipart form.</p></body></html>")

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
    try:
        img.save(save_path)
    except Exception:
        # ignore if saving fails; continue to predict
        pass

    try:
        if pipeline == "dl":
            # DL pipeline
            if not HAS_TF:
                return JSONResponse({"error": "DL pipeline unavailable: TensorFlow not installed."}, status_code=503)
            model = load_dl_model()
            inp = preprocess_for_dl(img)
            probs = model.predict(inp)[0]
            # load idx2label if possible
            ci_path = Path("model/class_indices.json")
            if ci_path.exists():
                try:
                    with open(ci_path) as f:
                        class_indices = json.load(f)
                        idx2label = {int(v): k for k, v in class_indices.items()}
                except Exception:
                    idx2label = {i: str(i) for i in range(len(probs))}
            else:
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
            # Classical pipeline
            if not HAS_JOBLIB:
                return JSONResponse({"error": "Classical pipeline unavailable: joblib not installed."}, status_code=503)
            clf, selector, extractor = load_classical_assets()
            inp = preprocess_for_extractor(img)
            # get embeddings using extractor (requires TF)
            if extractor is None:
                return JSONResponse({"error": "Embedding extractor not configured."}, status_code=500)
            emb = extractor.predict(inp)[0:1]  # shape (1,features)
            try:
                emb_sel = selector.transform(emb)
            except Exception as e:
                return JSONResponse({"error": f"Selector transform failed: {e}"}, status_code=500)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(emb_sel)[0]
                # prefer idx2label mapping if loaded
                idx2label = _idx2label if _idx2label is not None else None
                if idx2label is None:
                    try:
                        labels = clf.classes_
                        idx2label = {i: str(labels[i]) for i in range(len(labels))}
                    except Exception:
                        idx2label = {i: str(i) for i in range(len(probs))}
                topk = topk_from_probs(probs, idx2label, k=top_k)
            else:
                pred = clf.predict(emb_sel)[0]
                topk = [(str(pred), 1.0)]

            took = time.time() - start
            return JSONResponse({
                "pipeline": "classical",
                "predictions": [{"label": l, "confidence": p} for l,p in topk],
                "image_url": f"/static/uploads/{fname}",
                "took": took
            })

        else:
            return JSONResponse({"error": "Unknown pipeline. Use 'dl' or 'classical'."}, status_code=400)

    except FileNotFoundError as fe:
        return JSONResponse({"error": str(fe)}, status_code=500)
    except RuntimeError as re:
        return JSONResponse({"error": str(re)}, status_code=503)
    except Exception as e:
        # catch-all for unexpected errors
        return JSONResponse({"error": str(e)}, status_code=500)

# Static route to show the uploaded image files (FastAPI StaticFiles already mounted under /static)
@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    fpath = IMAGE_DIR / filename
    if fpath.exists():
        return FileResponse(str(fpath))
    return JSONResponse({"error": "file not found"}, status_code=404)

# If running locally you can use:
# uvicorn app:app --host 0.0.0.0 --port 8000

# app_classical.py - lightweight FastAPI for joblib classical models
import os, io, json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# optional huggingface download if you stored models there
from huggingface_hub import hf_hub_download

MODEL_DIR = "fs_saved"   # where your joblib models live
SELECTOR_PATH = os.path.join(MODEL_DIR, "variance_selector.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "SVM_variance_fs.joblib")  # adjust name if different

# Lazy imports for joblib/sklearn
HAS_JOBLIB = True
try:
    import joblib
except Exception as e:
    HAS_JOBLIB = False
    joblib = None
    print("Warning: joblib not installed:", e)

app = FastAPI(title="AgroAI (classical pipeline)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_emb(img: Image.Image):
    # lightweight embedding: use simple color histogram / flatten
    # OR, if you want to use the MobileNet extractor embeddings you created earlier,
    # you would need to download and load that extractor (heavier). Here we assume
    # your classical models accept the saved MobileNet embeddings as input (joblib pipeline saved).
    # If your saved classifiers expect embeddings, load the embeddings offline (preferred).
    img = img.convert("RGB").resize((224,224))
    arr = np.asarray(img).astype("float32") / 255.0
    # flatten color histogram (simple): change if classifier expects other features
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[:,:,ch], bins=128, range=(0,1))
        hist.extend(h.astype("float32"))
    return np.array(hist).reshape(1, -1)

# try to load joblib artifacts
SELECTOR = None
CLF = None
CLASS_INDICES = None
if HAS_JOBLIB:
    try:
        if os.path.exists(SVM_PATH):
            CLF = joblib.load(SVM_PATH)
            print("Loaded classifier:", SVM_PATH)
        else:
            print("Classifier file not found:", SVM_PATH)
        if os.path.exists(SELECTOR_PATH):
            SELECTOR = joblib.load(SELECTOR_PATH)
            print("Loaded selector:", SELECTOR_PATH)
    except Exception as e:
        print("Failed to load joblib artifacts:", e)
else:
    print("Joblib not available; classical pipeline disabled.")

@app.get("/", response_class=HTMLResponse)
async def root():
    html = """
    <html><body>
    <h3>AgroAI - Classical (SVM) service</h3>
    <form action="/predict" enctype="multipart/form-data" method="post">
      <input type="file" name="file" accept="image/*" />
      <input type="number" name="top_k" value="3" />
      <input type="submit" value="Predict"/>
    </form>
    <p>Use POST /predict for API calls.</p>
    </body></html>
    """
    return HTMLResponse(content=html)

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Form(3)):
    if not HAS_JOBLIB or CLF is None:
        return JSONResponse({"success": False, "error": "Classical pipeline not available (missing joblib/clf)."}, status_code=503)
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        feat = pil_to_emb(img)
        if SELECTOR is not None:
            feat = SELECTOR.transform(feat)
        if hasattr(CLF, "predict_proba"):
            probs = CLF.predict_proba(feat)[0]
            topi = probs.argsort()[-int(top_k):][::-1]
            # if classifier stores class labels as numbers, map to known names if available
            labels = getattr(CLF, "classes_", None)
            out = [{"label": str(labels[int(i)]) if labels is not None else str(int(i)), "score": float(probs[int(i)])} for i in topi]
        else:
            pred = CLF.predict(feat)[0]
            out = [{"label": str(pred), "score": 1.0}]
        return JSONResponse({"success": True, "predictions": out})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...), top_k: int = Form(3)):
    res = []
    for f in files:
        contents = await f.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        feat = pil_to_emb(img)
        if SELECTOR is not None:
            feat = SELECTOR.transform(feat)
        if hasattr(CLF, "predict_proba"):
            probs = CLF.predict_proba(feat)[0]
            topi = probs.argsort()[-int(top_k):][::-1]
            labels = getattr(CLF, "classes_", None)
            out = [{"label": str(labels[int(i)]) if labels is not None else str(int(i)), "score": float(probs[int(i)])} for i in topi]
        else:
            pred = CLF.predict(feat)[0]
            out = [{"label": str(pred), "score": 1.0}]
        res.append({"filename": f.filename, "predictions": out})
    return {"success": True, "results": res}

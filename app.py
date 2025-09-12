# app.py
import os, io, time, logging
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Optional Firebase
import firebase_admin
from firebase_admin import credentials, storage as fb_storage, firestore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-rec-api")

app = Flask(__name__)

# Config (env vars)
EMBED_PATH = os.environ.get("EMBEDDINGS_PATH", "embeddings.npy")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.npy")
FIREBASE_SECRET_FILE = os.environ.get("FIREBASE_SECRET_FILE", "/etc/secrets/serviceAccount.json")
FIREBASE_BUCKET = os.environ.get("hope-cff1b.firebasestorage.app", None)  # e.g. your-project-id.appspot.com
THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 0.7))

# Device & models (loaded once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Firebase init (if secret + bucket provided)
bucket = None
if os.path.exists(FIREBASE_SECRET_FILE) and FIREBASE_BUCKET:
    try:
        cred = credentials.Certificate(FIREBASE_SECRET_FILE)
        firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
        bucket = fb_storage.bucket()
        db = firestore.client()
        logger.info("Firebase initialized.")
    except Exception as e:
        logger.warning(f"Firebase init failed: {e}")
        bucket = None

# Helper: ensure embeddings & labels available (download from Firebase if needed)
def ensure_embeddings():
    global embeddings, labels
    if os.path.exists(EMBED_PATH) and os.path.exists(LABELS_PATH):
        logger.info("Loading embeddings & labels from local files.")
        embeddings = np.load(EMBED_PATH)
        labels = np.load(LABELS_PATH, allow_pickle=True)
    elif bucket:
        logger.info("Embeddings not found locally. Downloading from Firebase Storage...")
        tmp_embed = "/tmp/embeddings.npy"
        tmp_labels = "/tmp/labels.npy"
        try:
            bucket.blob("embeddings.npy").download_to_filename(tmp_embed)
            bucket.blob("labels.npy").download_to_filename(tmp_labels)
            embeddings = np.load(tmp_embed)
            labels = np.load(tmp_labels, allow_pickle=True)
            logger.info("Downloaded embeddings & labels from Firebase.")
        except Exception as e:
            logger.error("Failed to download embeddings from Firebase: " + str(e))
            embeddings = np.empty((0,512))
            labels = np.array([])
    else:
        logger.warning("No embeddings available. embeddings array empty.")
        embeddings = np.empty((0,512))
        labels = np.array([])

# Load on startup
ensure_embeddings()
if embeddings.size:
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    logger.info(f"Embeddings shape: {embeddings.shape}, labels: {len(labels)}")
else:
    logger.info("Embeddings is empty. Predictions will fail until embeddings are provided.")

# Utilities
def get_embedding_from_pil(img_pil):
    face = mtcnn(img_pil)
    if face is None:
        return None
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
    return emb.reshape(1, -1)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","time": int(time.time())})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error":"no image file provided"}), 400
    file = request.files['image']
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error":"invalid image", "details": str(e)}), 400

    emb = get_embedding_from_pil(img)
    if emb is None:
        return jsonify({"error":"no face detected"}), 400

    if embeddings.size == 0:
        return jsonify({"error":"no database embeddings available"}), 500

    sims = cosine_similarity(emb, embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_label = str(labels[best_idx])

    topk = int(request.form.get("topk", 3))
    idxs = sims.argsort()[::-1][:topk]
    results = [{"label": str(labels[i]), "score": float(sims[i])} for i in idxs]

    matched = best_score >= THRESHOLD
    resp = {
        "match": best_label if matched else None,
        "score": best_score,
        "threshold": THRESHOLD,
        "matched": bool(matched),
        "topk": results
    }
    return jsonify(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting app on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

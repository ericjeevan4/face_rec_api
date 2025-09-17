import os
import io
import time
import logging
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Model paths
MODEL_PATH = os.path.join("models", "vggface2_resnet.pth")

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load FaceNet ResNet model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    resnet = InceptionResnetV1(pretrained=None).eval()
    resnet.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    logging.info("✅ FaceNet model loaded successfully.")

except Exception as e:
    logging.error(f"❌ Error loading FaceNet model: {str(e)}")
    resnet = None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "ok", "time": int(time.time())})


@app.route("/recognize", methods=["POST"])
def recognize():
    """Face recognition API"""
    if resnet is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Detect faces
        faces, _ = mtcnn(img, return_prob=True)
        if faces is None:
            return jsonify({"error": "No face detected"}), 400

        # Generate embeddings
        embeddings = resnet(faces).detach().numpy()

        # Example similarity check (self-comparison)
        sim = cosine_similarity([embeddings[0]], [embeddings[0]])[0][0]

        return jsonify({"similarity": float(sim)})

    except Exception as e:
        logging.error(f"❌ Error in recognition: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

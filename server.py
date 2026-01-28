from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import torch
import torch.nn.functional as F  # <-- AJOUT
import torchvision.transforms as T
from PIL import Image
import io
import os
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

import socket  # <-- AJOUT SOCKET

# ==== PARAMÈTRES identiques au training ====
SR = 16000
DUR = 1.0                     # durée uniforme
N_MELS = 64
N_FFT = 512
HOP = 160
WIN = 400
OUT_SIZE = 224
CMAP = "magma"

# ---- AJOUT : Seuils "silence" + "confiance" ----
SILENCE_RMS_TH = 0.10   # à ajuster si besoin
CONF_TH = 0.90          # à ajuster si besoin

# ---- AJOUT : Paramètres socket Jetson + commandes autorisées ----
JETSON_IP = "192.168.4.1"   # <-- à remplacer
JETSON_PORT = 7002
ALLOWED_CMDS = {"left", "right", "go", "stop", "backward"}

# ==== IMPORTANT : Chemin de ton dossier IA ====
# BASE_DIR = r"C:\Users\gabys\OneDrive\Bureau\Paris\Sorbonne\Polytech\5a\IA embarqué\Version finale"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_model.pt")

# ==== Import de ton modèle ====
from resnet18 import ResNet, BasicBlock

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# 1) Charger le modèle IA
# ---------------------------------------------------
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
classes = checkpoint["classes"]

model = ResNet(
    img_channels=3,
    num_layers=18,
    block=BasicBlock,
    num_classes=len(classes)
)
model.load_state_dict(checkpoint["model"])
model.eval()

# Transforms pour transformer l'image en Tensor
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# ---------------------------------------------------
# Utilitaire : convertit un log-mel en image RGB 224x224
# ---------------------------------------------------
def mel_to_rgb(log_mel):
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis("off")
    librosa.display.specshow(log_mel, sr=SR, hop_length=HOP, cmap=CMAP)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    return img

# ---------------------------------------------------
# AJOUT : Envoi socket vers Jetson
# ---------------------------------------------------
def send_to_jetson(cmd: str):
    try:
        with socket.create_connection((JETSON_IP, JETSON_PORT), timeout=0.3) as s:
            s.sendall((cmd + "\n").encode("utf-8"))
            _ = s.recv(16)  # optionnel: lit "OK\n"
    except Exception as e:
        print("WARN socket Jetson:", e)


# ---------------------------------------------------
# 2) Route IA — reçoit un WAV brut envoyé depuis le HTML
# ---------------------------------------------------
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    try:
        # Récupérer base64 depuis JSON
        b64 = request.json["audio"]
        wav_bytes = base64.b64decode(b64)

        # Charger l'audio avec soundfile
        audio, sr = sf.read(io.BytesIO(wav_bytes))

        # Assurer que c'est mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample si nécessaire
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        # Pad ou cut pour obtenir DUR secondes
        target_len = int(SR * DUR)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        # ---- AJOUT : Détection silence (avant spectro) ----
        rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
        if rms < SILENCE_RMS_TH:
            return jsonify({"label": "mot non reconnu", "reason": "silence", "rms": rms})

        # ---- Création du Log-Mel spectrogramme identique au training ----
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SR,
            n_mels=N_MELS, n_fft=N_FFT,
            hop_length=HOP, win_length=WIN,
            power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # ---- Convertir en image RGB avec la colormap magma ----
        img = mel_to_rgb(log_mel)

        # ---- Convertir en Tensor ----
        tensor = transform(img).unsqueeze(0)

        # Debug :
        print("Tensor min/max :", tensor.min().item(), tensor.max().item())

        # ---- MODIF : Prédiction + confiance ----
        with torch.no_grad():
            out = model(tensor)               # logits
            probs = F.softmax(out, dim=1)[0]  # probas
            conf, pred = torch.max(probs, dim=0)

        conf = float(conf.item())
        pred = int(pred.item())

        if conf < CONF_TH:
            return jsonify({"label": "mot non reconnu", "confidence": conf})

        label = classes[pred]
        resp = {"label": label, "confidence": conf}

        # ---- AJOUT : envoyer uniquement les commandes reconnues ----
        if label in ALLOWED_CMDS:
            send_to_jetson(label)

        return jsonify(resp)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 400
    
# ---------------------------------------------------
# 3) AJOUT : Route commandes manuelles (boutons HTML)
# ---------------------------------------------------
@app.route("/send_cmd", methods=["POST"])
def send_cmd():
    try:
        cmd = request.json.get("cmd", "").strip().lower()

        if cmd not in ALLOWED_CMDS:
            return jsonify({"ok": False, "error": "cmd not allowed", "cmd": cmd}), 400

        send_to_jetson(cmd)
        return jsonify({"ok": True, "cmd": cmd})

    except Exception as e:
        print("ERROR /send_cmd:", e)
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/")
def home():
    return send_file("IHM.html")

# ---------------------------------------------------
# 4) Lancement du serveur
# ---------------------------------------------------
if __name__ == "__main__":
    print("🔥 Serveur IA Audio actif sur http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)





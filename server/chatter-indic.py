from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
import uuid
import re
import torch
import torchaudio as ta
import io
import numpy as np
import soundfile as sf
from datetime import timedelta
from transformers import AutoModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from fastapi.middleware.cors import CORSMiddleware
import tempfile

# 🔥 Firebase Init
cred = credentials.Certificate("./studio-7977682669-c03b0-firebase-adminsdk-fbsvc-78081c0376.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "studio-7977682669-c03b0.firebasestorage.app"
})

db = firestore.client()
bucket = storage.bucket()

# 🚀 Load Models (GPU)
indic_model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Request Model
class TTSRequest(BaseModel):
    text: str
    language_id: str
    audio_prompt_path: str
    ref_text: str = None  # only for Indic


# 🌍 Language Routing
INDIC_LANGS = {
    "as", "bn", "gu", "hi", "kn",
    "ml", "mr", "or", "pa", "ta", "te"
}

def choose_model(language_id: str):
    return "indic" if language_id in INDIC_LANGS else "chatterbox"


# 🔐 Verify Token
def verify_token(auth_header: str):
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing token")

    try:
        token = auth_header.split(" ")[1]
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# 🔐 Download audio
def download_audio(blob_path: str):
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)

    return buffer


# 🔹 Sentence splitter (for chatterbox)
def split_sentences(text):
    sentences = re.split(r'[।.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


# 🎤 Main Generator
def generate_audio(data: TTSRequest, uid: str):

    model_type = choose_model(data.language_id)

    audio_buffer = download_audio(data.audio_prompt_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_buffer.getvalue())
        tmp_path = tmp.name

    # =========================
    # 🇮🇳 INDICF5
    # =========================
    if model_type == "indic":

        if not data.ref_text:
            raise HTTPException(400, "ref_text required for IndicF5")

        audio = indic_model(
            data.text,
            ref_audio_path=tmp_path,
            ref_text=data.ref_text
        )

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        buffer = io.BytesIO()
        sf.write(buffer, np.array(audio, dtype=np.float32), 24000, format="WAV")
        buffer.seek(0)

    # =========================
    # 🌍 CHATTERBOX
    # =========================
    else:

        sentences = split_sentences(data.text)

        if len(sentences) == 0:
            raise HTTPException(400, "Empty text")

        all_wavs = []

        for sentence in sentences:
            with torch.no_grad():
                wav = chatterbox_model.generate(
                    sentence,
                    audio_prompt_path=tmp_path,
                    language_id=data.language_id,
                    temperature=0.4,
                    repetition_penalty=1.2,
                    cfg_weight=0.7,
                    exaggeration=0.6,
                    top_p=0.9
                )
            all_wavs.append(wav)

        final_wav = torch.cat(all_wavs, dim=-1)

        buffer = io.BytesIO()
        ta.save(buffer, final_wav, chatterbox_model.sr, format="wav")
        buffer.seek(0)

    # =========================
    # 📁 Upload
    # =========================
    file_name = f"{uuid.uuid4()}.wav"
    blob_path = f"users/{uid}/audios/{file_name}"

    blob = bucket.blob(blob_path)
    blob.upload_from_file(buffer, content_type="audio/wav")

    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=15),
        method="GET"
    )

    return signed_url, file_name, blob_path, model_type


# 🚀 Unified Endpoint
@app.post("/v1/tts")
async def unified_tts(data: TTSRequest, authorization: str = Header(None)):

    uid = verify_token(authorization)

    if len(data.text) > 4000:
        raise HTTPException(400, "Text too long")

    signed_url, file_name, blob_path, model_type = generate_audio(data, uid)

    # 💾 Save metadata
    db.collection("users").document(uid).collection("audios").add({
        "text": data.text,
        "language_id": data.language_id,
        "audio_prompt_path": data.audio_prompt_path,
        "ref_text": data.ref_text,
        "model_used": model_type,
        "file_name": file_name,
        "storage_path": blob_path,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    return {
        "message": "Audio generated",
        "model_used": model_type,
        "audio_url": signed_url
    }


# ✅ Health
@app.get("/")
def root():
    return {"message": "Unified TTS API 🚀"}

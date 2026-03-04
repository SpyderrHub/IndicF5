from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
import uuid
import io
import numpy as np
import soundfile as sf
from datetime import timedelta
from transformers import AutoModel
from fastapi.middleware.cors import CORSMiddleware
import tempfile

# 🔥 Firebase Init
cred = credentials.Certificate("./studio-7977682669-c03b0-firebase-adminsdk-fbsvc-78081c0376.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "studio-7977682669-c03b0.firebasestorage.app"
})

db = firestore.client()
bucket = storage.bucket()

# 🚀 Load IndicF5 model (GPU)
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

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
    ref_audio_path: str   # Firebase path
    ref_text: str


# 🔐 Verify Firebase Token
def verify_token(auth_header: str):
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing token")

    try:
        token = auth_header.split(" ")[1]
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# 🔐 Download reference audio
def download_audio(blob_path: str):
    try:
        blob = bucket.blob(blob_path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="Reference audio not found")

        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)

        return buffer

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🎤 Generate + Upload
def generate_and_upload(text, ref_audio_path, ref_text, uid):
    try:
        # 📥 Download reference audio
        audio_buffer = download_audio(ref_audio_path)

        # Save temp file (model needs path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_buffer.getvalue())
            tmp_path = tmp.name

        # 🎤 Generate audio
        audio = model(
            text,
            ref_audio_path=tmp_path,
            ref_text=ref_text
        )

        # 🔊 Normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # 🧠 Save to buffer
        buffer = io.BytesIO()
        sf.write(buffer, np.array(audio, dtype=np.float32), samplerate=24000, format="WAV")
        buffer.seek(0)

        # 📁 Upload to Firebase
        file_name = f"{uuid.uuid4()}.wav"
        blob_path = f"users/{uid}/audios/{file_name}"

        blob = bucket.blob(blob_path)
        blob.upload_from_file(buffer, content_type="audio/wav")

        # 🔐 Signed URL (15 min)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET"
        )

        return signed_url, file_name, blob_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


# 🚀 API Endpoint
@app.post("/v1/indic-tts")
async def indic_tts(data: TTSRequest, authorization: str = Header(None)):

    uid = verify_token(authorization)

    if len(data.text) > 4000:
        raise HTTPException(status_code=400, detail="Text too long")

    signed_url, file_name, blob_path = generate_and_upload(
        data.text,
        data.ref_audio_path,
        data.ref_text,
        uid
    )

    # 💾 Save metadata
    db.collection("users").document(uid).collection("audios").add({
        "text": data.text,
        "ref_text": data.ref_text,
        "ref_audio_path": data.ref_audio_path,
        "file_name": file_name,
        "storage_path": blob_path,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    return {
        "message": "IndicF5 audio generated",
        "audio_url": signed_url
    }


# ✅ Health Check
@app.get("/")
def root():
    return {"message": "IndicF5 API running 🚀"}

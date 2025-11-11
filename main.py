from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
from typing import Dict

app = FastAPI(
    title="BPM Detection API",
    description="Audio BPM detection service using Librosa",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "BPM Detection API",
        "version": "1.0.0"
    }


@app.post("/detect-bpm")
async def detect_bpm(file: UploadFile = File(...)) -> Dict:
    """
    Detect the BPM (Beats Per Minute) of an uploaded audio file.
    
    Supports common audio formats: MP3, WAV, FLAC, OGG, M4A, etc.
    
    Returns:
        - bpm: Detected tempo in beats per minute
        - filename: Original filename
        - confidence: Detection confidence score
    """
    
    # Validate file upload
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file extension
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Create a temporary file to store the upload
    temp_file = None
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Create temporary file with the same extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        # Load audio file with librosa
        y, sr = librosa.load(temp_file_path, sr=None)
        
        # Detect tempo using librosa's beat tracking
        # onset_envelope for more accurate detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Get tempo with aggregate method for better accuracy
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Alternative: Use dynamic programming for more accurate tempo estimation
        dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        
        # Calculate confidence based on variance of tempo estimates
        tempo_std = np.std(dtempo)
        confidence = max(0.0, min(1.0, 1.0 - (tempo_std / 50.0)))
        
        # Get the most accurate BPM using aggregate mean
        accurate_bpm = float(np.mean(dtempo))
        
        return {
            "bpm": round(accurate_bpm, 2),
            "filename": file.filename,
            "confidence": round(confidence, 2),
            "sample_rate": sr,
            "duration_seconds": round(len(y) / sr, 2)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


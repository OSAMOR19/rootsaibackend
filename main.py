from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BPM Detection API",
    description="Audio BPM detection service using Librosa",
    version="1.0.0"
)

# Configure CORS - Completely open, no restrictions whatsoever
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins - no restrictions
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],  # Allow ALL HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow ALL headers
    expose_headers=["*"],  # Expose ALL headers
    max_age=3600,  # Cache preflight requests for 1 hour
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
    temp_file_path = None
    try:
        logger.info(f"Processing file: {file.filename}, Content-Type: {file.content_type}")
        
        # Read the uploaded file
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Create temporary file with the same extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        logger.info(f"Temporary file created: {temp_file_path}")
        
        # Load audio file with librosa
        logger.info("Loading audio file with librosa...")
        y, sr = librosa.load(temp_file_path, sr=None, duration=None)
        logger.info(f"Audio loaded: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
        
        if len(y) == 0:
            raise HTTPException(status_code=400, detail="Audio file appears to be empty or corrupted")
        
        # Use multiple methods for more accurate BPM detection
        logger.info("Computing onset strength envelope...")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        
        # Method 1: Standard beat tracking (most reliable)
        logger.info("Method 1: Beat tracking...")
        tempo1, beats1 = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            units='tempo'
        )
        tempo1 = float(tempo1)
        logger.info(f"Method 1 result: {tempo1:.2f} BPM")
        
        # Method 2: Tempo estimation with dynamic programming
        logger.info("Method 2: Dynamic programming tempo estimation...")
        tempo2 = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            aggregate=np.median,
            start_bpm=60.0,
            std_bpm=1.0
        )
        tempo2 = float(tempo2)
        logger.info(f"Method 2 result: {tempo2:.2f} BPM")
        
        # Method 3: Tempo estimation with multiple candidates
        logger.info("Method 3: Multi-candidate tempo estimation...")
        tempo3_array = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            aggregate=None,
            start_bpm=60.0,
            std_bpm=1.0
        )
        # Filter out unrealistic values (between 60-200 BPM typically)
        tempo3_array = tempo3_array[(tempo3_array >= 60) & (tempo3_array <= 200)]
        if len(tempo3_array) > 0:
            tempo3 = float(np.median(tempo3_array))
            tempo_std = float(np.std(tempo3_array))
        else:
            tempo3 = tempo1  # Fallback to method 1
            tempo_std = 0.0
        logger.info(f"Method 3 result: {tempo3:.2f} BPM (std: {tempo_std:.2f})")
        
        # Combine results - use median of all methods for robustness
        tempos = [t for t in [tempo1, tempo2, tempo3] if 60 <= t <= 200]
        if not tempos:
            tempos = [tempo1]  # Fallback
        
        accurate_bpm = float(np.median(tempos))
        
        # Calculate confidence based on agreement between methods
        if len(tempos) >= 2:
            tempo_variance = np.var(tempos)
            confidence = max(0.0, min(1.0, 1.0 - (tempo_variance / 400.0)))
        else:
            confidence = 0.7  # Lower confidence if only one method
        
        # Ensure BPM is in reasonable range
        if accurate_bpm < 60:
            accurate_bpm *= 2  # Might be half-time
        elif accurate_bpm > 200:
            accurate_bpm /= 2  # Might be double-time
        
        logger.info(f"Final BPM: {accurate_bpm:.2f}, Confidence: {confidence:.2f}")
        
        result = {
            "bpm": round(accurate_bpm, 2),
            "filename": file.filename,
            "confidence": round(confidence, 2),
            "sample_rate": int(sr),
            "duration_seconds": round(len(y) / sr, 2),
            "method1_bpm": round(tempo1, 2),
            "method2_bpm": round(tempo2, 2),
            "method3_bpm": round(tempo3, 2)
        }
        
        logger.info(f"Returning result: {result}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


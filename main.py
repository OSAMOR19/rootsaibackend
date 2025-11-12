from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import logging
import soundfile as sf
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
        # Only process first 15 seconds for FAST BPM detection (enough for accurate results)
        # This prevents timeouts and ensures super fast processing
        max_duration = 15.0  # seconds
        logger.info(f"Loading audio file with librosa (max {max_duration}s for BPM detection)...")
        
        # Load with duration limit and resample to lower sample rate for SPEED
        y, sr = librosa.load(
            temp_file_path, 
            sr=11025,  # Lower sample rate = MUCH faster (still accurate for BPM)
            duration=max_duration,  # Only process first 15 seconds
            mono=True,  # Convert to mono for faster processing
            res_type='kaiser_fast'  # Faster resampling algorithm
        )
        
        actual_duration = len(y) / sr
        logger.info(f"Audio loaded: duration={actual_duration:.2f}s, sample_rate={sr}Hz")
        
        if len(y) == 0:
            raise HTTPException(status_code=400, detail="Audio file appears to be empty or corrupted")
        
        # Warn if file was truncated
        if actual_duration >= max_duration:
            logger.info(f"Note: Processing first {max_duration}s of audio for BPM detection (file is longer)")
        
        # Fast BPM detection using single method
        logger.info("Computing onset strength envelope...")
        # Use larger hop_length for faster processing
        hop_length = 1024  # Larger = faster (512 -> 1024 = 2x faster)
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr, 
            aggregate=np.median,
            hop_length=hop_length
        )
        logger.info(f"Onset envelope computed: {len(onset_env)} frames")
        
        # Standard beat tracking (most reliable and fastest)
        logger.info("Detecting BPM using beat tracking...")
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        bpm = float(tempo)
        logger.info(f"Detected BPM: {bpm:.2f}")
        
        # Simple result
        result = {
            "bpm": round(bpm, 2)
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


# BPM Detection API

A FastAPI backend service that accurately detects the BPM (Beats Per Minute) of uploaded audio files using Librosa.

## Features

- 🎵 Accurate BPM detection using Librosa's advanced algorithms
- 🚀 Fast and efficient audio processing
- 🌐 CORS enabled for Next.js frontend integration
- 🐳 Docker ready for Render deployment
- 📊 Returns confidence scores and audio metadata
- 🎼 Supports multiple audio formats (MP3, WAV, FLAC, OGG, M4A, etc.)

## API Endpoints

### `POST /detect-bpm`
Upload an audio file to detect its BPM.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Audio file (field name: `file`)

**Response:**
```json
{
  "bpm": 128.5,
  "filename": "song.mp3",
  "confidence": 0.95,
  "sample_rate": 22050,
  "duration_seconds": 180.5
}
```

### `GET /`
Health check and API information.

### `GET /health`
Simple health check for monitoring.

## Local Development

### Prerequisites
- Python 3.12 or 3.13 (recommended for compatibility)
  - **Note:** Python 3.14 is not yet supported by `numba` (a librosa dependency)
  - The Dockerfile uses Python 3.12 for deployment
- pip

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   # Using Python 3.12 (recommended for compatibility)
   python3.12 -m venv venv
   
   # Activate the virtual environment
   source venv/bin/activate
   
   # Or if using pyenv with Python 3.12:
   pyenv install 3.12.7
   pyenv local 3.12.7
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
```bash
# Make sure virtual environment is activated first
pip install -r requirements.txt
```

3. Run the server:
```bash
# Make sure virtual environment is activated
python main.py
```

Or with uvicorn directly:
```bash
# Make sure virtual environment is activated
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Note:** Always activate your virtual environment before running the server:
```bash
source venv/bin/activate
```

The API will be available at `http://localhost:8000`

### API Documentation
Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Deployment

### Build and run locally:
```bash
docker build -t bpm-detection-api .
docker run -p 8000:8000 bpm-detection-api
```

## Render Deployment

### Option 1: Deploy from GitHub

1. Push this code to a GitHub repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Render will auto-detect the Dockerfile
5. Deploy!

### Option 2: Manual Configuration

If Render doesn't auto-detect, use these settings:
- **Environment**: Docker
- **Build Command**: (leave empty, Dockerfile handles it)
- **Start Command**: (leave empty, Dockerfile handles it)
- **Port**: 8000

### Environment Variables (Optional)
You can set these in Render dashboard:
- `PORT`: 8000 (Render sets this automatically)
- `ALLOWED_ORIGINS`: Your Next.js frontend URL (for production CORS)

## Frontend Integration (Next.js)

Example fetch request:

```javascript
const detectBPM = async (audioFile) => {
  const formData = new FormData();
  formData.append('file', audioFile);
  
  const response = await fetch('https://your-api.onrender.com/detect-bpm', {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  console.log(`BPM: ${data.bpm}`);
  return data;
};
```

## Supported Audio Formats

- MP3
- WAV
- FLAC
- OGG
- M4A
- AIFF/AIF

## Technical Details

### BPM Detection Algorithm
The service uses Librosa's advanced tempo detection which:
1. Computes the onset strength envelope
2. Applies beat tracking algorithms
3. Uses dynamic programming for tempo estimation
4. Aggregates multiple tempo estimates for accuracy
5. Returns confidence score based on estimate variance

### Performance
- Average processing time: 2-5 seconds for a 3-minute song
- Memory efficient with temporary file cleanup
- Automatic sample rate optimization

## License

MIT License

## Support

For issues or questions, please open an issue on GitHub.


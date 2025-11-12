#!/usr/bin/env python3
"""
Simple test script for BPM detection API
Usage: python test_bpm.py <audio_file_path> [api_url]
"""

import sys
import requests
import json

def test_bpm_detection(audio_file_path, api_url="http://localhost:8000"):
    """Test the BPM detection endpoint"""
    
    endpoint = f"{api_url}/detect-bpm"
    
    print(f"Testing BPM detection API at: {endpoint}")
    print(f"Audio file: {audio_file_path}")
    print("-" * 50)
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': (audio_file_path, f, 'audio/mpeg')}
            
            print("Uploading file...")
            response = requests.post(endpoint, files=files)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ SUCCESS!")
                print(json.dumps(result, indent=2))
                print(f"\n🎵 Detected BPM: {result['bpm']}")
                print(f"📊 Confidence: {result['confidence']}")
                print(f"⏱️  Duration: {result['duration_seconds']}s")
                return True
            else:
                print(f"\n❌ ERROR: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except FileNotFoundError:
        print(f"❌ Error: File not found: {audio_file_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_bpm.py <audio_file_path> [api_url]")
        print("Example: python test_bpm.py song.mp3")
        print("Example: python test_bpm.py song.mp3 https://rootsaibackend.onrender.com")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    test_bpm_detection(audio_file, api_url)


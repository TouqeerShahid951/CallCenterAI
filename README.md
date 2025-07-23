# CallCenterAI

A real-time AI call center agent with voice interaction, live transcription, AI response generation, and speech output.

---

## Project Structure

```
CallCenterAI/
├── backend/
│   ├── src/
│   │   ├── main.py                # FastAPI backend, WebSocket audio streaming, VAD, Whisper, GODEL, TTS
│   │   └── controllers/
│   │       └── call_controller.py # (Your API routers, if any)
│   └── ...
├── webrtc-frontend/
│   └── src/
│       └── App.js                 # React frontend, mic capture, WebSocket, transcript UI
│   └── ...
└── README.md                      # This file
```

---

## Main Components

- **Backend (FastAPI, Python):**
  - `/ws/audio-stream` WebSocket endpoint for real-time audio streaming
  - Uses Silero VAD for speech detection
  - Whisper for speech-to-text (STT)
  - GODEL (HuggingFace) for LLM agent responses
  - gTTS for text-to-speech (TTS)
  - Trailing silence buffer for natural utterance segmentation

- **Frontend (React):**
  - Captures microphone audio, downsamples to 16kHz, frames into 30ms chunks
  - Streams audio to backend via WebSocket
  - Displays live transcript and agent responses
  - Plays agent TTS audio

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd CallCenterAI
```

### 2. Backend Setup
- **Python 3.8+ recommended**
- Create and activate a virtual environment:
  ```sh
  python -m venv virtenv
  source virtenv/bin/activate  # On Windows: .\virtenv\Scripts\activate
  ```
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  # If requirements.txt is missing, install manually:
  pip install fastapi uvicorn torch transformers gtts soundfile librosa numpy silero-vad
  ```
- Download models on first run (Whisper, GODEL)

- **Run the backend:**
  ```sh
  cd backend
  uvicorn src.main:app --reload
  # Or for production:
  uvicorn src.main:app --host 0.0.0.0 --port 8000
  ```

### 3. Frontend Setup
- **Node.js 16+ recommended**
- Install dependencies:
  ```sh
  cd webrtc-frontend
  npm install
  ```
- **Run the frontend:**
  ```sh
  npm start
  ```
- The app will be available at `http://localhost:3000`

---

## Usage
1. Open the frontend in your browser.
2. Click "Start Call" to begin streaming audio.
3. Speak into your microphone. The transcript and agent responses will appear in real time.
4. Click "End Call" to stop.

---

## Model Notes
- **Whisper**: Used for English speech-to-text. Model is loaded at backend startup.
- **GODEL**: Used for agent LLM responses. Model is loaded at backend startup.
- **Silero VAD**: Used for robust speech/silence detection.
- **gTTS**: Used for agent text-to-speech.

---

## Advanced
- **Trailing Silence Buffer**: Prevents splitting utterances on short pauses for more natural conversation.
- **Prompt Engineering**: GODEL prompt is tuned for customer support scenarios.
- **Debugging**: Each utterance is saved as `debug_utterance.wav` for backend inspection.

---

## Troubleshooting
- If you see model download errors, check your internet connection and Python version.
- If audio is not transcribed, check browser mic permissions and backend logs.
- For GPU acceleration, ensure PyTorch is installed with CUDA support.

---

## License
MIT (or your chosen license) 
# CallCenterAI Backend

A real-time AI call center backend built with FastAPI. Handles audio streaming, speech-to-text (Whisper), LLM agent responses (GODEL), and text-to-speech (gTTS).

---

## Features
- WebSocket endpoint for real-time audio streaming (`/ws/audio-stream`)
- Voice Activity Detection (Silero VAD)
- Speech-to-text with OpenAI Whisper
- LLM agent responses with GODEL (HuggingFace)
- Text-to-speech with gTTS
- Modular API routing

---

## Setup

### 1. Environment
- Python 3.8+
- (Recommended) Create a virtual environment:
  ```sh
  python -m venv virtenv
  # On Windows:
  .\virtenv\Scripts\Activate.ps1
  # On Linux/Mac:
  source virtenv/bin/activate
  ```

### 2. Install Dependencies
- From the `backend` directory:
  ```sh
  pip install -r requirements.txt
  ```

### 3. Run the Backend
- From the `backend` directory:
  ```sh
  uvicorn src.main:app --reload
  ```
- The API will be available at `http://127.0.0.1:8000`

---

## Endpoints
- `GET /test-audio` — Health check
- `POST /process-audio/` — (File upload, placeholder)
- `WS /ws/audio-stream` — Real-time audio streaming for live calls

---

## Tech Stack
- FastAPI (for backend)
- Uvicorn (ASGI server for building webapps, designed to handle websocets, HTTP)
- Whisper (STT)
- GODEL (LLM)
- gTTS (TTS)
- Silero VAD

---

## Notes
- Models are downloaded on first run.
- For GPU acceleration, install PyTorch with CUDA support.
- Debug utterances are saved as `debug_utterance.wav` for inspection.

---

## License
MIT (or your chosen license) 
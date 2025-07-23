from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.controllers import call_controller
import tempfile
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import numpy as np
import logging
import os
import json
import torch
import collections
import soundfile as sf

# Silero VAD via Torch Hub
logger = logging.getLogger(__name__)
logger.info("Loading Silero VAD model...")
try:
    vad_model, utils = torch.hub.load(
        'snakers4/silero-vad',
        'silero_vad',
        force_reload=False,
        onnx=False
    )
    vad_model.eval()  # Set to evaluation mode
    if torch.cuda.is_available():
        vad_model = vad_model.cuda()
    else:
        vad_model = vad_model.cpu()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    logger.info("✅ Silero VAD loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load Silero VAD: {str(e)}")
    raise

# GODEL setup
logger.info("Loading GODEL model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
logger.info("✅ GODEL loaded")

def generate_godel_response(instruction, knowledge, dialog):
    if knowledge:
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog_str = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog_str} {knowledge}"
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(call_controller.router)

# Load Whisper model
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("base.en")
logger.info("✅ Whisper loaded")

logging.basicConfig(level=logging.INFO)

@app.get("/test-audio")
async def test_audio():
    return {"message": "Audio test endpoint working", "status": "ready"}

@app.websocket("/ws/audio-stream")
async def websocket_audio_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔄 WebSocket connection accepted")

    # State management
    buffer = b""  # Incoming audio bytes buffer
    ring_buffer = collections.deque(maxlen=int(300/30))  # 300ms lookback buffer
    utterance_frames = []  # Frames that make up the current utterance
    speech_count = silence_count = 0  # Consecutive frame counters
    triggered = False  # VAD state
    conversation_history = []  # Dialog history
    should_cancel = False  # Interruption flag
    
    # Constants
    FRAME_MS = 30  # Process audio in 30ms frames
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS/1000)
    FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit audio = 2 bytes per sample
    VAD_BUFFER_MS = 500  # VAD rolling buffer size in ms
    VAD_BUFFER_SAMPLES = int(SAMPLE_RATE * VAD_BUFFER_MS / 1000)
    START_THRESHOLD = 3  # Number of speech frames to trigger start (90ms)
    END_THRESHOLD = 6    # Number of silence frames to trigger end (180ms)
    MAX_UTTERANCE_FRAMES = 100  # ~3 seconds max utterance
    TRAILING_SILENCE_FRAMES = 10  # 300ms trailing buffer after end-of-speech
    try:
        logger.info("🎤 Starting audio processing loop...")
        vad_buffer = np.zeros(VAD_BUFFER_SAMPLES, dtype=np.float32)
        trailing_buffer = []  # Buffer for trailing silence frames
        in_trailing = False   # Are we in the trailing silence window?
        trailing_count = 0
        while True:
            try:
                msg = await websocket.receive()
            except RuntimeError as e:
                logger.info(f"WebSocket disconnect detected: {e}")
                break

            # Handle text messages (e.g., cancel)
            if msg.get("type") == "text":
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "cancel":
                        logger.info("🚫 Cancel received - resetting state")
                        should_cancel = True
                        # Reset VAD state
                        buffer = b""
                        utterance_frames.clear()
                        speech_count = silence_count = 0
                        triggered = False
                        continue
                except Exception as e:
                    logger.warning(f"Failed to parse text message: {e}")
                    continue

            # Handle audio data
            if msg.get("bytes"):
                buffer += msg["bytes"]
                logger.info(f"📥 Received audio chunk: {len(msg['bytes'])} bytes")

            # Process frames
            while len(buffer) >= FRAME_BYTES:
                logger.info(f"Buffer size before frame processing: {len(buffer)}")
                # Extract one frame
                frame = buffer[:FRAME_BYTES]
                buffer = buffer[FRAME_BYTES:]
                
                # Convert to float32 [-1,1] for VAD
                audio_np = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0
                
                # Rolling buffer for VAD (500ms)
                vad_buffer = np.roll(vad_buffer, -FRAME_SAMPLES)
                vad_buffer[-FRAME_SAMPLES:] = audio_np

                # Run VAD on the rolling buffer
                speech_timestamps = get_speech_timestamps(vad_buffer, vad_model, sampling_rate=SAMPLE_RATE)
                logger.info(f"speech_timestamps: {speech_timestamps}")

                # Are we in speech? (last timestamp ends within last frame)
                is_speech = False
                if speech_timestamps:
                    last = speech_timestamps[-1]
                    if last['end'] > len(vad_buffer) - FRAME_SAMPLES:
                        is_speech = True
                logger.info(f"VAD is_speech: {is_speech}, speech_count: {speech_count}, silence_count: {silence_count}")
                
                # Update speech/silence counters
                if is_speech:
                    speech_count += 1
                    silence_count = 0
                    logger.debug(f"🗣️ Speech frame detected (count: {speech_count})")
                else:
                    silence_count += 1
                    speech_count = 0
                    logger.debug(f"🤫 Silence frame detected (count: {silence_count})")

                # --- Trailing silence logic ---
                if in_trailing:
                    trailing_buffer.append(frame)
                    trailing_count += 1
                    if is_speech:
                        # Speech resumed, merge trailing buffer and continue
                        utterance_frames.extend(trailing_buffer)
                        trailing_buffer.clear()
                        in_trailing = False
                        trailing_count = 0
                        continue  # Go back to normal speech state
                    elif trailing_count >= TRAILING_SILENCE_FRAMES:
                        # Trailing silence expired, process utterance
                        utterance_frames.extend(trailing_buffer)
                        trailing_buffer.clear()
                        in_trailing = False
                        trailing_count = 0
                        # --- process utterance as before ---
                        pcm = b"".join(utterance_frames)
                        audio_float = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
                        sf.write("debug_utterance.wav", audio_float, SAMPLE_RATE, subtype='PCM_16')
                        logger.info("🤖 Transcribing with Whisper...")
                        result = whisper_model.transcribe(
                            audio_float, 
                            language="en",
                            fp16=False
                        )
                        transcript = result.get("text", "").strip()
                        logger.info(f"📝 Transcript: '{transcript}'")
                        await websocket.send_json({"type": "transcript", "text": transcript})
                        if transcript and len(transcript) > 2:
                            conversation_history.append({"role": "user", "content": transcript})
                            if len(conversation_history) == 1:
                                response = "Hello! I'm your AI assistant. How can I help you today?"
                                logger.info("👋 Using welcome message")
                            else:
                                instruction = (
                                    "Instruction: You are a helpful, knowledgeable, and concise customer support agent for Acme Corp. "
                                    "Always answer user questions factually, politely, and clearly. If you do not know the answer, say so honestly. "
                                    "Do not make up information. If the user asks about company policy, returns, or math, answer as best as you can."
                                )
                                dialog = [t["content"] for t in conversation_history[-4:]]
                                logger.info("🧠 Generating GODEL response...")
                                response = generate_godel_response(instruction, "", dialog)
                            conversation_history.append({"role": "agent", "content": response})
                            logger.info(f"🤖 Agent response: '{response}'")
                            await websocket.send_json({"type": "agent_response", "text": response})
                            logger.info("🔊 Generating TTS...")
                            tts = gTTS(response, lang='en', slow=False)
                            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                                tts.write_to_fp(f)
                                path = f.name
                            with open(path, 'rb') as f:
                                audio_bytes = f.read()
                                await websocket.send_bytes(audio_bytes)
                            os.remove(path)
                            logger.info("✅ TTS audio sent")
                        utterance_frames.clear()
                        speech_count = silence_count = 0
                        triggered = False
                        should_cancel = False
                        continue  # Go to next frame
                    else:
                        continue  # Still in trailing silence

                # --- Normal state machine ---
                if not triggered and speech_count >= START_THRESHOLD:
                    triggered = True
                    logger.info("🟢 Start of speech detected")
                    for f, _ in ring_buffer:
                        utterance_frames.append(f)
                    utterance_frames.append(frame)
                elif triggered:
                    utterance_frames.append(frame)
                    if silence_count >= END_THRESHOLD or \
                       len(utterance_frames) >= MAX_UTTERANCE_FRAMES or \
                       should_cancel:
                        logger.info("🔴 End of speech detected - entering trailing silence buffer")
                        in_trailing = True
                        trailing_buffer.clear()
                        trailing_count = 0
                        continue  # Start trailing silence window
                ring_buffer.append((frame, is_speech))
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected by client")
    except Exception as e:
        logger.exception("❌ WebSocket error")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass  # Ignore if already closed
    
    logger.info("🏁 Connection closed") 
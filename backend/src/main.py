from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.controllers import call_controller
import asyncio
import base64
import tempfile
import whisper
from transformers import pipeline
from gtts import gTTS
import numpy as np
import logging
import os
import soundfile as sf
import librosa

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(call_controller.router)

# Load models once
whisper_model = whisper.load_model("base.en")
llm = pipeline("text-generation", model="microsoft/DialoGPT-small")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/test-audio")
async def test_audio():
    return {"message": "Audio test endpoint working", "status": "ready"}

@app.websocket("/ws/audio-stream")
async def websocket_audio_stream_endpoint(websocket: WebSocket):
    logger.info("🔄 New WebSocket connection request received")
    await websocket.accept()
    logger.info("✅ WebSocket connection accepted")
    
    audio_buffer = b""
    conversation_history = []
    is_processing = False
    target_sample_rate = 16000
    bytes_per_sample = 2  # int16
    target_duration = 3  # seconds
    required_bytes = target_sample_rate * bytes_per_sample * target_duration
    
    try:
        logger.info("🎤 Starting audio processing loop...")
        while True:
            try:
                # Receive audio chunk from frontend
                data = await websocket.receive_bytes()
                audio_buffer += data
                logger.info(f"📥 Received audio chunk: {len(data)} bytes, total buffer: {len(audio_buffer)} bytes")
                
                # Process audio when we have enough data (dynamically calculated for 3 seconds)
                if len(audio_buffer) > required_bytes and not is_processing:
                    is_processing = True
                    logger.info(f"🎯 Processing audio buffer: {len(audio_buffer)} bytes")
                    
                    try:
                        # Convert buffer to numpy array
                        pcm_data = np.frombuffer(audio_buffer, dtype=np.int16)
                        logger.info(f"🔢 Converted to PCM array: {len(pcm_data)} samples")
                        
                        # Check if audio has meaningful content (not just silence)
                        audio_volume = np.abs(pcm_data).mean()
                        logger.info(f"🔊 Audio volume: {audio_volume}")
                        
                        if audio_volume < 50:  # Very low volume - likely silence
                            logger.info("🔇 Detected silence, skipping processing")
                            audio_buffer = b""
                            is_processing = False
                            continue
                        
                        # Assume mono 16kHz PCM from frontend, write directly to WAV
                        mono_data = pcm_data
                        logger.info(f"🎵 Using mono audio: {len(mono_data)} samples at 16kHz")
                        
                        # Create temporary WAV file for Whisper
                        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                        os.close(fd)
                        sf.write(tmp_path, mono_data, target_sample_rate, subtype='PCM_16')
                        logger.info(f"💾 Created temp WAV file: {tmp_path}")
                        
                        logger.info("🤖 Starting Whisper transcription...")
                        # Transcribe with Whisper
                        result = whisper_model.transcribe(tmp_path, language="en", fp16=False)
                        transcript = result["text"].strip()
                        
                        # Clean up temp file
                        os.remove(tmp_path)
                        logger.info(f"🗑️ Deleted temp WAV file")
                        
                        logger.info(f"📝 Whisper transcript: '{transcript}'")
                        # Send transcript to frontend for display
                        await websocket.send_json({"type": "transcript", "text": transcript})
                        
                        if transcript and len(transcript) > 2:  # Only process if we got a meaningful transcript
                            conversation_history.append({"role": "user", "content": transcript})
                            logger.info(f"💬 Added to conversation history. Total turns: {len(conversation_history)}")
                            
                            # Generate AI response with improved prompt
                            if len(conversation_history) == 1:
                                agent_response = "Hello! I'm your AI assistant. How can I help you today?"
                                logger.info("👋 Using welcome message for first interaction")
                            else:
                                prompt = f"User: {transcript}\nAgent:"
                                logger.info(f"🧠 Generating response with prompt: {prompt}")
                                response = llm(prompt, max_length=100, do_sample=True, temperature=0.8, truncation=True)
                                generated = response[0]["generated_text"].strip()
                                logger.info(f"🤖 Raw AI response: '{generated}'")
                                # Extract agent response
                                if generated.startswith(prompt):
                                    agent_response = generated[len(prompt):].strip()
                                else:
                                    agent_response = generated
                                if not agent_response or len(agent_response) < 3:
                                    agent_response = "I understand. Please tell me more about that."
                                    logger.info("⚠️ Using fallback response")
                            conversation_history.append({"role": "agent", "content": agent_response})
                            logger.info(f"🤖 Final AI Agent response: '{agent_response}'")
                            logger.info("🗣️ Generating TTS audio...")
                            # Generate TTS audio
                            tts = gTTS(agent_response, lang='en', slow=False)
                            fd, tts_file_path = tempfile.mkstemp(suffix=".mp3")
                            with os.fdopen(fd, "wb") as tts_file:
                                tts.write_to_fp(tts_file)
                            logger.info(f"💾 Created TTS MP3 file: {tts_file_path}")
                            # Read TTS audio and send back
                            with open(tts_file_path, "rb") as tts_file:
                                tts_audio_bytes = tts_file.read()
                            # Clean up TTS file
                            os.remove(tts_file_path)
                            logger.info(f"🗑️ Deleted TTS MP3 file")
                            logger.info(f"📡 Sending TTS audio: {len(tts_audio_bytes)} bytes")
                            # Send TTS audio back to frontend
                            await websocket.send_bytes(tts_audio_bytes)
                            logger.info(f"✅ TTS audio sent: {len(tts_audio_bytes)} bytes")
                        else:
                            logger.info("⚠️ No meaningful transcript, skipping response")
                        # Reset buffer for next chunk
                        audio_buffer = b""
                        logger.info("🔄 Reset audio buffer for next chunk")
                        is_processing = False
                    except Exception as processing_error:
                        logger.exception(f"❌ Audio processing error: {processing_error}")
                        import traceback
                        traceback.print_exc()
                        audio_buffer = b""  # Reset buffer on error
                        is_processing = False
            except Exception as receive_error:
                logger.exception(f"❌ Error receiving data: {receive_error}")
                break
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected by client")
    except Exception as e:
        logger.exception(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close(code=1011, reason=str(e))
    logger.info("🏁 WebSocket connection ended") 
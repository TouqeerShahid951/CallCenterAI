import time
import logging
import tempfile
import soundfile as sf
from transformers import pipeline
from gtts import gTTS
import base64
import whisper
from fastapi import HTTPException

# Load Whisper model once at module level
model = whisper.load_model("base")
# Load conversational LLM pipeline
llm = pipeline("text-generation", model="microsoft/DialoGPT-small")

async def process_audio(file):
    try:
        timings = {}
        start = time.perf_counter()
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        # Whisper STT
        stt_start = time.perf_counter()
        result = model.transcribe(tmp_path)
        transcript = result["text"]
        timings["stt_ms"] = (time.perf_counter() - stt_start) * 1000
        # LLM response
        llm_start = time.perf_counter()
        response = llm(transcript, max_length=100)
        agent_response = response[0]["generated_text"]
        timings["llm_ms"] = (time.perf_counter() - llm_start) * 1000
        # TTS with gTTS
        tts_start = time.perf_counter()
        tts = gTTS(agent_response)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
            tts.write_to_fp(tts_file)
            tts_file.seek(0)
            tts_audio_bytes = tts_file.read()
        tts_audio_b64 = base64.b64encode(tts_audio_bytes).decode("utf-8")
        timings["tts_ms"] = (time.perf_counter() - tts_start) * 1000
        total = (time.perf_counter() - start) * 1000
        logging.info(f"process_audio executed in {total:.2f} ms (STT: {timings['stt_ms']:.2f} ms, LLM: {timings['llm_ms']:.2f} ms, TTS: {timings['tts_ms']:.2f} ms)")
        return {
            "transcript": transcript,
            "agent_response": agent_response,
            "tts_audio_b64": tts_audio_b64,
            "timings": timings,
            "total_latency_ms": total
        }
    except Exception as e:
        logging.exception("Error in process_audio")
        raise HTTPException(status_code=500, detail={"error": str(e)}) 
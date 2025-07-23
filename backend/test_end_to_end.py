import whisper
from gtts import gTTS

# 1. Transcribe
model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
transcript = result["text"]
print("Transcript:", transcript)

# 2. TTS
tts = gTTS(transcript or "This is a test.")
tts.save("test_tts_from_transcript.mp3")
print("TTS audio saved: test_tts_from_transcript.mp3") 
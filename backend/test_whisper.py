import whisper
import os

# Optional: record from mic if no file is present
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

def record_from_mic(filename="mic_test.wav", duration=5, samplerate=16000):
    if sd is None or sf is None:
        print("sounddevice and soundfile are required for mic recording. Run: pip install sounddevice soundfile")
        return False
    print(f"Recording {duration} seconds from mic...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"Saved recording to {filename}")
    return True

# Main logic
AUDIO_FILE = "audio.mp3"
if not os.path.exists(AUDIO_FILE):
    AUDIO_FILE = "mic_test.wav"
    if not os.path.exists(AUDIO_FILE):
        recorded = record_from_mic(AUDIO_FILE, duration=5, samplerate=16000)
        if not recorded:
            exit(1)

model = whisper.load_model("small.en")
result = model.transcribe(AUDIO_FILE)
print("Transcription:", result["text"]) 
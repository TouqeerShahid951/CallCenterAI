import numpy as np
import soundfile as sf

# Simulate 1 second of silence at 16kHz, 16-bit PCM
pcm_data = np.zeros(16000, dtype=np.int16)
sf.write("test_pcm_to_wav.wav", pcm_data, 16000, subtype='PCM_16')
print("WAV file written: test_pcm_to_wav.wav") 
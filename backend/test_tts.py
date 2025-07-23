from gtts import gTTS

text = "Hello, this is a test of the text to madaeboard speech system."
tts = gTTS(text)
tts.save("test_output.mp3")
print("TTS audio saved as test_output.mp3")

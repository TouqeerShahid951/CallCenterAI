from gtts import gTTS

test_text = "This is a test of the text-to-speech system."
tts = gTTS(test_text, lang='en', slow=False)
tts.save("test_tts.mp3")
print("TTS audio saved as test_tts.mp3")

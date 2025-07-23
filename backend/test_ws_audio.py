import asyncio
import websockets

async def send_audio():
    uri = "ws://127.0.0.1:8000/ws/audio-stream"
    async with websockets.connect(uri) as websocket:
        with open("mic_test.wav", "rb") as f:
            data = f.read()
            await websocket.send(data)
        print("Audio sent. Waiting for responses...")
        tts_audio = b""
        try:
            while True:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                if isinstance(msg, bytes):
                    print(f"Received binary audio data: {len(msg)} bytes. Saving as test_ws_tts.mp3.")
                    tts_audio += msg
                    with open("test_ws_tts.mp3", "wb") as out:
                        out.write(tts_audio)
                else:
                    print(f"Received text message: {msg}")
        except asyncio.TimeoutError:
            print("No more messages from backend (timeout). Test complete.")

if __name__ == "__main__":
    asyncio.run(send_audio()) 
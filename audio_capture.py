import sounddevice as sd
import numpy as np
import queue

SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_SECONDS = 8
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata.copy())

def record_chunk():
    buffer = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        sd.sleep(CHUNK_SECONDS * 1000)
    while not q.empty():
        buffer.append(q.get())
    return np.concatenate(buffer, axis=0)

if __name__ == "__main__":
    print("Recording 8 seconds... speak now.")
    audio = record_chunk()
    print(f"Captured audio shape: {audio.shape}")
import whisper
import numpy as np
from audio_capture import record_chunk

print("Loading Whisper model...")
model = whisper.load_model("small")

print("Recording 8 seconds... speak clearly now.")
audio = record_chunk()
audio = audio.flatten().astype(np.float32)

print("Transcribing...")
result = model.transcribe(audio, fp16=False, language="en")

print("Transcript:")
print(result["text"])
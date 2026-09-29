import sounddevice as sd
import numpy as np
import queue
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
import time

SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
q = queue.Queue()
transcript_buffer = ""
latest_summary = "No summary yet."

# ---- Audio capture ----
def callback(indata, frames, time_, status):
    q.put(indata.copy())

def record_chunk():
    buffer = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        sd.sleep(CHUNK_SECONDS * 1000)
    while not q.empty():
        buffer.append(q.get())
    return np.concatenate(buffer, axis=0)

# ---- Load models once ----
print("Loading Whisper...")
whisper_model = whisper.load_model("small")

print("Loading summarization model (float16 - lighter on RAM)...")
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

def summarize(text):
    prompt = f"""Summarize this meeting transcript in 2-3 bullet points. Then list any action items.

Transcript:
{text}

Summary:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---- Main loop ----
def main_loop():
    global transcript_buffer, latest_summary
    chunk_count = 0
    while True:
        print("\nListening...")
        audio = record_chunk()
        audio = audio.flatten().astype(np.float32)
        result = whisper_model.transcribe(audio, fp16=False, language="en")
        text = result["text"].strip()
        if text:
            transcript_buffer += " " + text
            print(f"Transcript so far:\n{transcript_buffer}\n")

        chunk_count += 1
        if chunk_count % 3 == 0:  # summarize every ~24 seconds
            print("Updating summary...")
            latest_summary = summarize(transcript_buffer)
            print(f"Summary:\n{latest_summary}\n")

if __name__ == "__main__":
    main_loop()
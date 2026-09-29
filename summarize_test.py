from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/Phi-3.5-mini-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (this will take a while and download ~7GB the first time)...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

def summarize(transcript_text):
    prompt = f"""Summarize this meeting transcript in 2-3 bullet points. Then list any action items mentioned.

Transcript:
{transcript_text}

Summary:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_transcript = "We need to finish the design mockups by Friday. Sarah will handle the client email. Let's push the launch to next month."
    print("Summarizing...")
    print(summarize(sample_transcript))
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = os.environ.get("LLAMA_MODEL_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../deployed_model")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

class Message(BaseModel):
    messages: list  # [{"role": "user"/"system"/"assistant", "content": "..."}]
    temperature: float = 0.7
    max_tokens: int = 512

print(f"Loading model from {MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

@app.post("/v1/chat/completions")
def chat_completion(msg: Message):
    # Simple chat template: concatenate all messages
    prompt = ""
    for m in msg.messages:
        if m["role"] == "system":
            prompt += f"[SYSTEM] {m['content']}\n"
        elif m["role"] == "user":
            prompt += f"[USER] {m['content']}\n"
        elif m["role"] == "assistant":
            prompt += f"[ASSISTANT] {m['content']}\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=msg.max_tokens,
            temperature=msg.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return {
        "choices": [{
            "message": {"role": "assistant", "content": generated}
        }]
    }

@app.get("/")
def root():
    return {
        "status": "llama-3.1-8B finetuned API ready",
        "test_curl": "curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{"role":"user","content":"Xin chào, bạn là ai?"}]}'",
        "swagger": "/docs",
        "prompt_example": [
            {"role": "system", "content": "Bạn là trợ lý AI pháp luật."},
            {"role": "user", "content": "Xin chào, bạn là ai?"}
        ]
    }

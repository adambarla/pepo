#!/usr/bin/env python3
"""
Simple OpenAI-compatible API server using transformers + FastAPI
This serves as a lightweight alternative to vLLM
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import time

# Configuration
MODEL_NAME = "PessimisticDPO/SmolLM2-1.7Bdpo_ensemble_with_1.0alpha1_l0"
PORT = 8001
HOST = "0.0.0.0"

# Load model and tokenizer
print(f"Loading model: {MODEL_NAME}")

# Get HuggingFace token from environment
import os
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN not found in environment. Model may fail to load if private.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,  # Fixed: use 'dtype' instead of deprecated 'torch_dtype'
    device_map="auto",
    trust_remote_code=True,
    token=hf_token
)
print("Model loaded successfully!")

# FastAPI app
app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    # Format messages into a prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": msg.role, "content": msg.content} for msg in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Return OpenAI-compatible response
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop"
        }]
    )

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    print(f"\nStarting server at {HOST}:{PORT}")
    print(f"API endpoint: http://{HOST}:{PORT}/v1/chat/completions")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=True)

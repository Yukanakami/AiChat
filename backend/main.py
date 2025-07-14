from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.post("/chat")
async def chat(req: ChatRequest):
    user_input = req.message
    print("User:", user_input)

    if not user_input.strip():
        return {"response": "Tolong ketik sesuatu."}

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')


    output_ids = model.generate(
    input_ids,
    max_length=100,
    pad_token_id=tokenizer.eos_token_id  
)

    reply = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    print("Bot:", reply)
    return {"response": reply}

# app/models/request_models.py

from pydantic import BaseModel

class TextGenRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0

class TextGenResponse(BaseModel):
    prompt: str
    generated_text: str
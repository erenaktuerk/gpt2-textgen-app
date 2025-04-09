# app/routes/textgen_routes.py

from fastapi import APIRouter
from app.models.request_models import TextGenRequest, TextGenResponse
from app.services.text_generator import GPT2TextGenerator

router = APIRouter()
generator = GPT2TextGenerator()

@router.post("/", response_model=TextGenResponse)
def generate_text(request: TextGenRequest):
    """
    Generate text using a pre-trained GPT-2 model.
    """
    output_text = generator.generate_text(
        prompt=request.prompt,
        max_length=request.max_length,
        temperature=request.temperature
    )
    return TextGenResponse(prompt=request.prompt, generated_text=output_text)
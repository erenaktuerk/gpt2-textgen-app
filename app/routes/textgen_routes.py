from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.text_generator import GPT2TextGenerator

router = APIRouter()

# Definiere das Request-Modell
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0

# Initialisiere den Textgenerator
generator = GPT2TextGenerator()

@router.post("/textgen/")
async def generate_text(request: TextGenerationRequest):
    # Fehlerbehandlung: Prüfen, ob der Prompt leer ist
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    try:
        # Text generieren
        generated_text = generator.generate_text(
            request.prompt, max_length=request.max_length, temperature=request.temperature
        )
        return {"prompt": request.prompt, "generated_text": generated_text}
    
    except Exception as e:
        # Fehlerbehandlung für unerwartete Fehler
        raise HTTPException(status_code=500, detail=f"An error occurred during text generation: {str(e)}")
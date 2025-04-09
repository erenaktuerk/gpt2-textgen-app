# app/main.py

from fastapi import FastAPI
from app.routes.textgen_routes import router as textgen_router

app = FastAPI(
    title="GPT-2 Text Generation API",
    description="Generate text using a pre-trained GPT-2 model.",
    version="1.0.0"
)

# Include the router for text generation
app.include_router(textgen_router, prefix="/api/v1/textgen", tags=["Text Generation"])
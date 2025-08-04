from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="Vocal Analysis API", version="1.0.0")

# CORS para n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "name": "Vocal Analysis API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "home": "/",
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/analyze")
async def analyze_audio(data: dict):
    """
    Endpoint para análise de áudio.
    Por enquanto retorna dados mock para teste.
    """
    return {
        "status": "success",
        "pitch_analysis": {
            "mean_frequency_hz": 440.0,
            "note": "A4",
            "range": "A3-A5"
        },
        "vocal_classification": {
            "type": "Soprano",
            "confidence": 0.75
        },
        "message": "API funcionando! Deploy Railway bem-sucedido."
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Vocal Analysis API"}

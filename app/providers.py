import os
from typing import TypedDict, List, Dict, Optional

# --- METADATA DEL MODELO ---

class ModelCapability(TypedDict):
    hasThinking: bool
    canSearch: bool
    isMultimodal: bool
    maxOutputTokens: int
    temperature: float
    isHidden: Optional[bool]

MODEL_METADATA: Dict[str, ModelCapability] = {
    # Lógica y Razonamiento (Thinking Models)
    "openai/gpt-oss-120b:free": { "hasThinking": True, "canSearch": True, "isMultimodal": False, "maxOutputTokens": 32768, "temperature": 0.6, "isHidden": False },
    "openai/gpt-oss-20b:free":  { "hasThinking": True, "canSearch": True, "isMultimodal": False, "maxOutputTokens": 16384, "temperature": 0.7, "isHidden": False },
    "gemini/gemini-3-flash-preview": { "hasThinking": True, "canSearch": True, "isMultimodal": True, "maxOutputTokens": 8192, "temperature": 0.7, "isHidden": False },
    
    # Conversación y Versatilidad (Groq / Cerebras)
    "groq/llama-3.3-70b-versatile": { "hasThinking": False, "canSearch": True, "isMultimodal": False, "maxOutputTokens": 4096, "temperature": 0.7, "isHidden": False },
    "groq/llama-3.1-8b-instant":    { "hasThinking": False, "canSearch": True, "isMultimodal": False, "maxOutputTokens": 4096, "temperature": 0.7, "isHidden": False },
    "cerebras/llama3.1-70b":        { "hasThinking": False, "canSearch": True, "isMultimodal": False, "maxOutputTokens": 8192, "temperature": 0.5, "isHidden": False },
    
    # Visión y Auxiliares (Ocultos por defecto)
    "google/gemma-3-4b-it:free":   { "hasThinking": False, "canSearch": False, "isMultimodal": True, "maxOutputTokens": 8192, "temperature": 0.7, "isHidden": True },
    "google/gemma-3-12b-it:free":  { "hasThinking": False, "canSearch": False, "isMultimodal": True, "maxOutputTokens": 8192, "temperature": 0.7, "isHidden": True },
    "google/gemma-3-27b-it:free":  { "hasThinking": False, "canSearch": False, "isMultimodal": True, "maxOutputTokens": 8192, "temperature": 0.7, "isHidden": True },
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free": { "hasThinking": False, "canSearch": True, "isMultimodal": False, "maxOutputTokens": 8192, "temperature": 0.7, "isHidden": True },
    
    # Generación de Imagen
    "olvera-image-1.0": { "hasThinking": False, "canSearch": False, "isMultimodal": False, "maxOutputTokens": 4096, "temperature": 1.0, "isHidden": False },
}

MODEL_LABELS: Dict[str, str] = {
    "openai/gpt-oss-120b:free": "GPT OSS 120B (OpenRouter)",
    "openai/gpt-oss-20b:free":  "GPT OSS 20B (OpenRouter)",
    "gemini/gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "groq/llama-3.3-70b-versatile": "Olvera AI (Llama 3.3 70B)",
    "groq/llama-3.1-8b-instant":    "Olvera AI (Llama 3.1 8B)",
    "cerebras/llama3.1-70b":        "Olvera AI (Cerebras Ultra-Fast)",
    "google/gemma-3-4b-it:free":    "Gemma 3 4B Vision",
    "google/gemma-3-12b-it:free":   "Gemma 3 12B Vision",
    "google/gemma-3-27b-it:free":   "Gemma 3 27B Vision",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free": "Dolphin Mistral 24B",
    "olvera-image-1.0": "Olvera Image 1.0 (Flux)",
}

DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"

def get_visible_models() -> List[str]:
    return [m for m, meta in MODEL_METADATA.items() if not meta.get("isHidden", False)]

def get_model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)

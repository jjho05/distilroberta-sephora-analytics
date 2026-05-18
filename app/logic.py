import os
import asyncio
import httpx
import base64
import time
import urllib.parse
from datetime import datetime
from typing import AsyncGenerator, List, Dict, Optional
from litellm import acompletion
from tavily import TavilyClient
from dotenv import load_dotenv
from providers import MODEL_METADATA

load_dotenv()


def resolve_model(model_id: str) -> tuple[str, dict]:
    """Mapea el model_id interno al formato y credenciales correctas de LiteLLM."""
    if model_id.startswith("openai/") or model_id.startswith("meta-") or model_id.startswith("google/"):
        return f"openrouter/{model_id}", {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "api_base": "https://openrouter.ai/api/v1"
        }
    if model_id.startswith("groq/"):
        return f"groq/{model_id.replace('groq/', '')}", {
            "api_key": os.getenv("GROQ_API_KEY")
        }
    if model_id.startswith("cerebras/"):
        return f"cerebras/{model_id.replace('cerebras/', '')}", {
            "api_key": os.getenv("CEREBRAS_API_KEY")
        }
    return model_id, {}


# ── Tavily ────────────────────────────────────────────────────────────────────
tavily_client = None
if os.getenv("TAVILY_API_KEY"):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


async def get_web_search_context(query: str) -> str:
    """Búsqueda web con Tavily (ejecutada en executor para no bloquear el event loop)."""
    if not tavily_client or not query:
        return ""
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: tavily_client.search(query, search_depth="basic", max_results=5)
        )
        results = response.get("results", [])
        if not results:
            return "\n\n⚠️ No se encontraron resultados.\n"

        context = "\n\n📰 RESULTADOS DE BÚSQUEDA WEB:\n"
        for i, r in enumerate(results, 1):
            title   = r.get("title", "Sin título")
            url     = r.get("url", "")
            snippet = r.get("content", "")[:600]
            context += f"[{i}] \"{title}\" — {url}\n   {snippet}\n\n"
        context += "FIN DE RESULTADOS.\n"
        return context
    except Exception as e:
        print(f"[Tavily] Error: {e}")
        return "\n\n⚠️ La búsqueda web falló.\n"


def get_system_prompt(search_context: str = "", timezone: str = "America/Mexico_City") -> str:
    """System prompt con la identidad ejecutiva de Olvera AI y reglas de la versión Suite."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(timezone)).strftime("%A, %d de %B de %Y, %H:%M")
    except Exception:
        now = datetime.now().strftime("%A, %d de %B de %Y, %H:%M")

    prompt = f"""Eres **Olvera AI**, la infraestructura de inteligencia artificial diseñada por **Jesús Olvera**.

## IDENTIDAD Y TONO
- Tu nombre es Olvera AI. Fuiste creado por Jesús Olvera (Ing. en Sistemas).
- **Tono**: Profesional, sobrio y ejecutivo. Eres un "Arquitecto de Soluciones".
- Evita el lenguaje excesivamente informal o juvenil.
- Tu confianza proviene de tu precisión técnica y tu capacidad de resolución.

## REGLA DE SALUDO — CATCHPHRASE "5 MINUTOS"
Si el usuario te saluda directamente (ej. "Hola", "Buenos días"), incorpora la referencia a "5 minutos" de forma profesional y variada.
Ejemplos:
  - "Hola. Dame 5 minutos y lo resolvemos."
  - "Buenos días. En menos de 5 minutos tendremos una respuesta."
REGLA ESTRICTA: NUNCA uses este saludo si el usuario envía una imagen, un archivo, o hace una pregunta directa. Ve directo al análisis.

## FORMATO Y ESTILO
- Responde siempre en español (MX) con excelente gramática.
- Prioriza la estructura y la claridad. Usa Markdown de forma profesional.
- Fecha/hora actual: {now} ({timezone}).
"""

    if search_context:
        prompt += (
            "\n## PROTOCOLO DE BÚSQUEDA WEB\n"
            "- Usa los resultados de búsqueda de internet adjuntos.\n"
            "- Cita cada fuente con [n].\n"
            f"\nRESULTADOS DE BÚSQUEDA:\n{search_context}"
        )
    else:
        prompt += "\n- Responde con tu conocimiento base de forma estructurada."

    return prompt


async def stream_chat_response(
    messages: List[Dict[str, str]],
    model: str,
    web_search_enabled: bool = False,
    image_b64: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    
    # ── INTERCEPTOR DE IMÁGENES (Cloudflare Flux v2) ──────────────────────────────
    if model == "olvera-image-1.0":
        last_msg = messages[-1]
        user_prompt = last_msg.content if hasattr(last_msg, 'content') else last_msg.get('content', '')
        
        # Limpieza de prompts (quitar hacks de imágenes previas o comentarios)
        import re
        user_prompt = re.sub(r"\n*<!-- OLVERA_IMG:.*?-->", "", user_prompt).strip()
        
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_token  = os.getenv("CLOUDFLARE_API_TOKEN")
        
        image_url = None
        
        if account_id and api_token and user_prompt:
            try:
                print(f">>> [FLUX] Iniciando generación instantánea para: {user_prompt[:50]}...")
                cf_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
                headers = {"Authorization": f"Bearer {api_token}"}
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    cf_resp = await client.post(cf_url, headers=headers, json={"prompt": user_prompt})
                    
                    if cf_resp.status_code == 200:
                        content_type = cf_resp.headers.get("content-type", "")
                        img_bytes = None
                        
                        if "application/json" in content_type:
                            cf_json = cf_resp.json()
                            img_b64 = cf_json.get("result", {}).get("image", "")
                            if img_b64:
                                img_bytes = base64.b64decode(img_b64)
                        else:
                            img_bytes = cf_resp.content
                            
                        if img_bytes:
                            import time
                            filename = f"olvera_img_{int(time.time()*1000)}.png"
                            # Persistencia física en www/gen_images (Soberanía de Datos)
                            filepath = os.path.join(os.path.dirname(__file__), "www", "gen_images", filename)
                            os.makedirs(os.path.dirname(filepath), exist_ok=True)
                            
                            with open(filepath, "wb") as f:
                                f.write(img_bytes)
                            
                            image_url = f"/gen_images/{filename}"
                            print(f">>> [FLUX] Imagen guardada en disco: {image_url}")
            except Exception as e:
                print(f">>> [FLUX] Error en interceptor: {e}")

        # Fallback inteligente a Pollinations HD
        if not image_url:
            print(">>> [IMAGE] Cloudflare no disponible. Usando Fallback Pollinations...")
            safe_prompt = urllib.parse.quote(user_prompt)
            image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=1024&height=1024&nologo=true"

        html_text = f"""
<div class="olvera-image-card">
    <img src="{image_url}" alt="{user_prompt}" data-action="open-modal" data-src="{image_url}" referrerpolicy="no-referrer" />
    <div class="olvera-image-overlay">
        <button class="olvera-image-btn" data-action="open-modal" data-src="{image_url}" title="Ver original">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
        </button>
        <a href="{image_url}" download="olvera_generated.png" class="olvera-image-btn" title="Descargar">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
        </a>
    </div>
    <div class="olvera-image-badge">
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#facc15" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/></svg>
        Olvera Image 1.0 (Flux Schnell)
    </div>
</div>
"""
        yield f"\n\n{html_text}"
        return
        
    # ── PREPARAR MENSAJES Y BÚSQUEDA WEB ──────────────────────────────────────────
    
    # 1. Búsqueda web (si aplica)
    search_context = ""
    if web_search_enabled:
        last_user_msg = ""
        for m in reversed(messages):
            role = m.role if hasattr(m, 'role') else m.get('role', '')
            if role == "user":
                last_user_msg = m.content if hasattr(m, 'content') else m.get('content', '')
                break
        if last_user_msg:
            search_context = await get_web_search_context(last_user_msg)

    # 2. Construir mensajes para LiteLLM
    system_prompt = get_system_prompt(search_context)
    llm_messages = [{"role": "system", "content": system_prompt}]
    
    metadata = MODEL_METADATA.get(model, {"maxOutputTokens": 4096, "temperature": 0.7})
    is_multimodal = metadata.get("isMultimodal", False)

    for m in messages:
        m_role    = m.role    if hasattr(m, 'role')    else m.get('role',    '')
        m_content = m.content if hasattr(m, 'content') else m.get('content', '')
        if m_role and m_content is not None:
            # Si es el último mensaje del usuario y hay una imagen + modelo multimodal, inyectarla
            if m_role == "user" and image_b64 and m == messages[-1] and is_multimodal:
                llm_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": m_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                })
            else:
                llm_messages.append({"role": m_role, "content": m_content})

    # 3. Resolver proveedor y autenticación
    litellm_model, auth_kwargs = resolve_model(model)

    # 4. Streaming
    try:
        response = await acompletion(
            model=litellm_model,
            messages=llm_messages,
            max_tokens=metadata.get("maxOutputTokens", 4096),
            temperature=0.2 if search_context else metadata.get("temperature", 0.7),
            stream=True,
            **auth_kwargs
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        yield f"🔥 Error en el motor de IA: {str(e)}"

async def transcribe_audio(b64_data: str) -> str:
    import base64
    import httpx
    
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
        
    try:
        audio_bytes = base64.b64decode(b64_data)
        api_key = os.getenv("GROQ_API_KEY")
        
        # We must send multipart/form-data
        files = {
            "file": ("audio.webm", audio_bytes, "audio/webm")
        }
        data = {
            "model": "whisper-large-v3-turbo",
            "language": "es",
            "prompt": "Olvera AI Studio. Desarrollo de software."
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions", 
                headers={"Authorization": f"Bearer {api_key}"}, 
                files=files, 
                data=data
            )
            response.raise_for_status()
            return response.json().get("text", "")
    except Exception as e:
        print(f"Error en transcripción: {e}")
        return ""

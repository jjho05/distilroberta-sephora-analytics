"""
history.py - Gestión de historial de chats para Olvera AI Shiny
Emula el hook useHistory() del Suite de Next.js
"""
from shiny import reactive
from datetime import datetime
from typing import List, Dict, Optional
import json

# ── Estado global reactivo ───────────────────────────────────────────────────
_chat_store: reactive.Value[List[Dict]] = reactive.Value([])

def get_history() -> List[Dict]:
    return _chat_store.get()

def add_chat(chat_id: str, title: str, messages: List[Dict]) -> None:
    store = _chat_store.get().copy()
    store.insert(0, {
        "id": chat_id,
        "title": title,
        "messages": messages,
        "created_at": datetime.now().strftime("%H:%M")
    })
    _chat_store.set(store)

def update_chat(chat_id: str, messages: List[Dict]) -> None:
    store = _chat_store.get().copy()
    for c in store:
        if c["id"] == chat_id:
            c["messages"] = messages
            break
    _chat_store.set(store)

def update_chat_title(chat_id: str, new_title: str) -> None:
    store = _chat_store.get().copy()
    for c in store:
        if c["id"] == chat_id:
            c["title"] = new_title
            break
    _chat_store.set(store)

def delete_chat(chat_id: str) -> None:
    store = [c for c in _chat_store.get() if c["id"] != chat_id]
    _chat_store.set(store)

def get_chat(chat_id: str) -> Optional[Dict]:
    return next((c for c in _chat_store.get() if c["id"] == chat_id), None)

def auto_title(messages: List[Dict]) -> str:
    """Genera un título automático del primer mensaje del usuario."""
    for m in messages:
        role = m.role if hasattr(m, 'role') else m.get('role', '')
        content = m.content if hasattr(m, 'content') else m.get('content', '')
        if role == 'user' and content:
            return content[:40] + ("..." if len(content) > 40 else "")
    return "Nueva consulta"

from shiny.express import ui, input, render
from shiny import reactive, session
import time
import base64
from pathlib import Path
import logic
import providers
import history as hist

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. OPTIMIZACIÓN Y CARGA DE DATOS (SEPHORA)
# ==========================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "reviews_emociones_opt.csv")
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")
    df["year"] = df["submission_time"].dt.year
    diccionario_es = {'anger': 'Ira', 'disgust': 'Desagrado', 'fear': 'Miedo', 'joy': 'Alegría', 'neutral': 'Neutral', 'sadness': 'Tristeza', 'surprise': 'Sorpresa'}
    df["emotion_es"] = df["emotion"].map(diccionario_es).fillna('Neutral')

    sent_by_day = df.groupby([df["submission_time"].dt.to_period("M"), "emotion_es"]).size().reset_index(name="count")
    sent_by_day["date"] = sent_by_day["submission_time"].dt.to_timestamp()
    sent_totals = df.groupby("emotion_es").size().reset_index(name="count")
    heatmap_df = df.groupby(["year", "emotion_es"]).size().reset_index(name="count")
    rating_df = df.groupby("emotion_es")["rating"].mean().reset_index(name="avg_rating")
    brand_totals = df.groupby(["brand_name", "emotion_es"]).size().reset_index(name="count")
    top_brands = df["brand_name"].value_counts().nlargest(10).index
    brand_df = brand_totals[brand_totals["brand_name"].isin(top_brands)]
    cat_df = df.groupby(["primary_category", "emotion_es"]).size().reset_index(name="count")
else:
    df = pd.DataFrame()

COLOR_PALETTE = { 'Alegría': '#D4AF37', 'Desagrado': '#FF5733', 'Ira': '#FF0055', 'Miedo': '#8E44AD', 'Neutral': '#7F8C8D', 'Tristeza': '#3498DB', 'Sorpresa': '#2ECC71' }


current_chat_id = reactive.Value(None)
is_empty        = reactive.Value(True)
uploaded_file_data = reactive.Value(None)

# ── LIMPIEZA DE CACHÉ ──────────────────────────────────────────────
# Limpia las imágenes generadas anteriormente al iniciar para no acumular basura.
def cleanup_gen_images():
    import shutil
    import os
    gen_path = os.path.join(os.path.dirname(__file__), "www", "gen_images")
    if os.path.exists(gen_path):
        try:
            # Opcional: Solo borrar archivos de más de 24h, o borrar todo al inicio.
            shutil.rmtree(gen_path)
            os.makedirs(gen_path, exist_ok=True)
        except Exception as e:
            print(f"Error en limpieza de imágenes: {e}")
    else:
        os.makedirs(gen_path, exist_ok=True)

cleanup_gen_images()

ui.page_opts(
    title=None,
    fillable=True,
    window_title="Olvera AI",
    lang="es",
    static_assets=str(Path(__file__).parent / "www")
)
ui.include_css("www/style.css")

ui.head_content(ui.HTML("""
<meta name="referrer" content="no-referrer">
<script src="mic.js"></script>
<script>
// ── Manejo Global de Imágenes (Delegación para evitar sanitización) ──────────
function openImageModal(src) {
    if (!src) return;
    const modal = document.createElement('div');
    modal.id = 'olvera-image-modal';
    modal.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(0,0,0,0.85);backdrop-filter:blur(10px);cursor:zoom-out;padding:20px;';
    
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'position:relative;max-width:90%;max-height:90%;display:flex;align-items:center;justify-content:center;cursor:default;';

    const img = document.createElement('img');
    img.src = src;
    img.style.cssText = 'max-width:100%;max-height:100%;border-radius:12px;box-shadow:0 25px 50px -12px rgba(0,0,0,0.5);object-fit:contain;';
    
    const controls = document.createElement('div');
    controls.style.cssText = 'position:absolute;top:-50px;right:0;display:flex;gap:12px;';

    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
    closeBtn.style.cssText = 'background:rgba(255,255,255,0.15);border:none;color:white;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;backdrop-filter:blur(8px);transition:all 0.2s;';
    closeBtn.onclick = () => document.body.removeChild(modal);

    const downloadBtn = document.createElement('button');
    downloadBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>';
    downloadBtn.style.cssText = 'background:rgba(255,255,255,0.15);border:none;color:white;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;backdrop-filter:blur(8px);transition:all 0.2s;';
    downloadBtn.title = 'Descargar';
    downloadBtn.onclick = (e) => { e.stopPropagation(); downloadImage(src); };

    [closeBtn, downloadBtn].forEach(b => {
        b.onmouseover = () => b.style.background = 'rgba(255,255,255,0.25)';
        b.onmouseout = () => b.style.background = 'rgba(255,255,255,0.15)';
    });

    modal.onclick = (e) => { if (e.target === modal) document.body.removeChild(modal); };
    
    controls.appendChild(downloadBtn);
    controls.appendChild(closeBtn);
    wrapper.appendChild(img);
    wrapper.appendChild(controls);
    modal.appendChild(wrapper);
    document.body.appendChild(modal);
}

async function downloadImage(src) {
    if (!src) return;
    try {
        // Usar fetch para obtener el blob y forzar la descarga correctamente (evita que el navegador solo abra la imagen)
        const response = await fetch(src);
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `olvera-ai-${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        
        // Limpieza
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 100);
    } catch (e) {
        // Fallback simple si fetch falla (ej. CORS)
        const a = document.createElement('a');
        a.href = src;
        a.download = `olvera-ai-${Date.now()}.png`;
        a.target = '_blank';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

document.addEventListener('click', (e) => {
    // 1. Detectar clic en imagen o botón con data-action="open-modal"
    const modalTrigger = e.target.closest('[data-action="open-modal"]');
    // 2. O detectar cualquier imagen dentro de un mensaje de chat
    const chatImg = e.target.closest('.shiny-chat-message img, .chat-img-preview');
    
    if (modalTrigger || chatImg) {
        const target = modalTrigger || chatImg;
        const src = target.getAttribute('data-src') || target.src;
        if (src) openImageModal(src);
        return;
    }
    
    // Detectar clic en botón de descarga
    const downloadTrigger = e.target.closest('[data-action="download-image"]');
    if (downloadTrigger) {
        const src = downloadTrigger.getAttribute('data-src');
        if (src) downloadImage(src);
        return;
    }
});

// ── Sidebar toggle ────────────────────────────────────────────────────────────
function toggleSidebar() {
    const sidebar = document.getElementById('olvera-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const isOpen  = sidebar.classList.contains('sidebar-open');
    if (isOpen) {
        sidebar.classList.remove('sidebar-open');
        overlay.classList.remove('overlay-visible');
    } else {
        sidebar.classList.add('sidebar-open');
        overlay.classList.add('overlay-visible');
    }
}

// ── Toggle búsqueda web ───────────────────────────────────────────────────────
function toggleWebSearch() {
    const btn = document.getElementById('web-search-btn');
    const isOn = btn.classList.toggle('toolbar-btn-active');
    Shiny.setInputValue('web_search', isOn, {priority: 'event'});
}

// ── Modelo selector ───────────────────────────────────────────────────────────
function toggleModelDropdown() {
    const dd = document.getElementById('model-dropdown');
    dd.classList.toggle('model-dd-open');
}

function selectModel(modelId, label, provider) {
    document.getElementById('model-label').textContent    = label;
    document.getElementById('model-provider').textContent = provider;
    Shiny.setInputValue('selected_model', modelId, {priority: 'event'});
    document.getElementById('model-dropdown').classList.remove('model-dd-open');
}

document.addEventListener('click', function(e) {
    const dd      = document.getElementById('model-dropdown');
    const trigger = document.getElementById('model-selector-btn');
    if (dd && trigger && !trigger.contains(e.target) && !dd.contains(e.target)) {
        dd.classList.remove('model-dd-open');
    }
    const cdd      = document.getElementById('config-dropdown');
    const ctrigger = document.getElementById('config-btn');
    if (cdd && ctrigger && !ctrigger.contains(e.target) && !cdd.contains(e.target)) {
        cdd.classList.remove('config-dd-open');
    }
});

// ── Tema ──────────────────────────────────────────────────────────────────────
function toggleConfigDropdown() {
    const dd = document.getElementById('config-dropdown');
    dd.classList.toggle('config-dd-open');
}

function setTheme(theme) {
    const html = document.documentElement;
    if (theme === 'system') {
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        html.setAttribute('data-theme', isDark ? 'dark' : 'light');
        localStorage.removeItem('olvera-theme');
    } else {
        html.setAttribute('data-theme', theme);
        localStorage.setItem('olvera-theme', theme);
    }
    updateThemeUI();
    document.getElementById('config-dropdown').classList.remove('config-dd-open');
}

function updateThemeUI() {
    const saved = localStorage.getItem('olvera-theme') || 'system';
    document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('theme-btn-active'));
    const el = document.getElementById('theme-' + saved);
    if (el) el.classList.add('theme-btn-active');
}

(function() {
    const saved = localStorage.getItem('olvera-theme');
    const html  = document.documentElement;
    if (saved) {
        html.setAttribute('data-theme', saved);
    } else {
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        html.setAttribute('data-theme', isDark ? 'dark' : 'light');
    }
    window.addEventListener('DOMContentLoaded', updateThemeUI);
})();

// ── CHAT BAR ──────────────────────────────────────────────────────────────────
function sendMessage() {
    const ta = document.getElementById('olvera-textarea');
    if (!ta) return;
    const text = ta.value.trim();
    if (!text) return;

    const origTA  = document.querySelector('shiny-chat-input textarea');
    const origBtn = document.querySelector('shiny-chat-input button.shiny-chat-btn-send');

    if (origTA && origBtn) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
        setter.call(origTA, text);
        origTA.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: text }));
        origTA.dispatchEvent(new Event('change', { bubbles: true }));
        setTimeout(() => { origBtn.removeAttribute('disabled'); origBtn.click(); }, 50);
    } else {
        Shiny.setInputValue('chat_user_input', text, { priority: 'event' });
    }

    ta.value = '';
    ta.style.height = 'auto';
    ta.focus();
}

document.addEventListener('DOMContentLoaded', function() {
    const ta = document.getElementById('olvera-textarea');
    if (!ta) return;

    ta.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    ta.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // ── SOPORTE DE PEGADO (PASTE IMAGES) ─────────────────────────────────────
    ta.addEventListener('paste', function(e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let index in items) {
            const item = items[index];
            if (item.kind === 'file' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                const reader = new FileReader();
                reader.onload = function(event) {
                    // Enviamos como si fuera una subida normal pero indicando que es base64
                    Shiny.setInputValue('file_pasted', {
                        name: 'pasted_image.png',
                        data: event.target.result.split(',')[1]
                    }, {priority:'event'});
                };
                reader.readAsDataURL(blob);
            }
        }
    });

    // ── ESTADO DEL BOTÓN DE ENVÍO ─────────────────────────────────────────────
    const sendBtn = document.getElementById('toolbar-send-btn');
    function updateSendBtnState() {
        const text = ta.value.trim();
        const hasFile = document.getElementById('file-indicator').style.display === 'block';
        sendBtn.disabled = (!text && !hasFile);
        sendBtn.style.opacity = (text || hasFile) ? '1' : '0.4';
        sendBtn.style.cursor = (text || hasFile) ? 'pointer' : 'not-allowed';
    }
    ta.addEventListener('input', updateSendBtnState);
    
    // Escuchar cambios en el indicador de archivo
    const observer = new MutationObserver(updateSendBtnState);
    observer.observe(document.getElementById('file-indicator'), { attributes: true, attributeFilter: ['style'] });
    
    updateSendBtnState(); // Estado inicial
});

Shiny.addCustomMessageHandler('file_status', function(message) {
    console.log('File Status Received:', message);
    const area = document.getElementById('attachment-preview-area');
    const dot = document.getElementById('file-indicator');
    
    if (dot) dot.style.display = message.active ? 'block' : 'none';
    
    if (!message.active) {
        if (area) area.innerHTML = '';
        const input = document.getElementById('file_upload_input');
        if (input) input.value = '';
    } else {
        if (area) {
            // Mostrar imagen si hay preview, si no mostrar icono de archivo
            const content = message.type === 'image' && message.preview
                ? `<img src="${message.preview}" style="width:100%;height:100%;object-fit:cover;" />`
                : `<div class="file-icon" style="color:var(--brand-color);"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg></div>`;
            
            area.innerHTML = `
                <div class="attachment-chip">
                    ${content}
                    <div class="attachment-remove-btn" onclick="Shiny.setInputValue('remove_attachment', true, {priority:'event'})">×</div>
                </div>
            `;
        }
    }
    if (typeof updateSendBtnState === 'function') updateSendBtnState();
});
</script>
"""))

# ── TOP BAR ───────────────────────────────────────────────────────────────────
ui.HTML("""
<header class="olvera-topbar">
    <div class="topbar-left">
        <button class="topbar-icon-btn" onclick="toggleSidebar()" title="Menú">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="3" y1="6"  x2="21" y2="6"/>
                <line x1="3" y1="12" x2="21" y2="12"/>
                <line x1="3" y1="18" x2="21" y2="18"/>
            </svg>
        </button>
        <div class="topbar-brand">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:18px;height:18px;">
                <path d="M12 2L3 7V17L12 22L21 17V7L12 2Z"/>
                <path d="M12 22V12"/><path d="M12 12L21 7"/><path d="M12 12L3 7"/>
            </svg>
            OLVERA AI
        </div>
    </div>
    <div class="topbar-config-area">
        <button id="config-btn" class="topbar-icon-btn" onclick="toggleConfigDropdown()" title="Configuración">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
        </button>
        <div id="config-dropdown" class="config-dropdown">
            <div class="config-section">
                <div class="config-label">Tema</div>
                <div class="theme-options">
                    <button class="theme-btn" onclick="setTheme('light')" id="theme-light">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
                        Claro
                    </button>
                    <button class="theme-btn" onclick="setTheme('dark')" id="theme-dark">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
                        Oscuro
                    </button>
                    <button class="theme-btn" onclick="setTheme('system')" id="theme-system">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                        Sistema
                    </button>
                </div>
            </div>
        </div>
    </div>
</header>
""")

# ── SIDEBAR + OVERLAY ─────────────────────────────────────────────────────────
ui.HTML('<div id="sidebar-overlay" class="sidebar-overlay" onclick="toggleSidebar()"></div>')

with ui.div(id="olvera-sidebar", class_="olvera-sidebar"):
    ui.HTML("""
        <div class="sidebar-brand">
            <div class="brand-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:18px;height:18px;">
                    <path d="M12 2L3 7V17L12 22L21 17V7L12 2Z"/>
                    <path d="M12 22V12"/><path d="M12 12L21 7"/><path d="M12 12L3 7"/>
                </svg>
            </div>
            <div>
                <div class="brand-name">Olvera AI</div>
                <div class="brand-sub">Programacion para Ciencia de Datos</div>
            </div>
        </div>
    """)
    ui.input_action_button("new_chat", "＋  Nuevo Chat", class_="btn-new-chat")

    ui.h3("Sephora Analytics", style="color: #000; font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 0.95rem; text-align: center; margin-top: 10px;")
    ui.input_checkbox_group("emociones", "FILTRAR EMOCIONES:", choices=sorted(list(diccionario_es.values())), selected=sorted(list(diccionario_es.values())))

    ui.hr()
    ui.div("🕘 Historial", class_="sidebar-section-title")
    ui.input_text("search_history", None, placeholder="Buscar historial...", width="100%")

    @render.ui
    def chat_history_ui():
        all_chats = hist.get_history()
        query     = (input.search_history() or "").lower()
        active    = current_chat_id.get()
        chats     = [c for c in all_chats if query in c["title"].lower()] if query else all_chats
        if not chats:
            return ui.div(ui.HTML('<div class="history-empty">Sin registros</div>'))
        items = []
        for c in chats:
            is_active = c["id"] == active
            items.append(ui.div(
                ui.div(
                    ui.span(c["title"], class_="history-title"),
                    ui.span(c["created_at"], class_="history-time"),
                    class_="history-meta"
                ),
                ui.div(
                    ui.HTML(
                        f'<button class="history-action" '
                        f'onclick="Shiny.setInputValue(\'edit_chat_id\', \'{c["id"]}\', {{priority:\'event\'}});event.stopPropagation();"'
                        f'title="Editar"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>'
                        f'<button class="history-action" '
                        f'onclick="Shiny.setInputValue(\'delete_chat_id\', \'{c["id"]}\', {{priority:\'event\'}});event.stopPropagation();"'
                        f'title="Eliminar"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button>'
                    ),
                    class_="history-actions"
                ),
                class_=f"history-item {'history-item-active' if is_active else ''}",
                onclick=f"Shiny.setInputValue('load_chat_id', '{c['id']}', {{priority:'event'}})"
            ))
        return ui.div(*items, class_="history-list")

    ui.hr()
    ui.div("© 2026 Olvera AI Lab", class_="sidebar-footer")

# ── ÁREA PRINCIPAL ────────────────────────────────────────────────────────────
with ui.div(class_="main-area html-fill-item html-fill-container"):

    with ui.navset_tab(id="main_navset"):
        with ui.nav_panel("💬 Olvera AI Copilot"):

            @render.ui
            def welcome_ui():
                if not is_empty.get():
                    return ui.div()
                return ui.div(
                    ui.div(
                        ui.HTML("""
                            <div class="welcome-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width:36px;height:36px;">
                                    <path d="M12 2L3 7V17L12 22L21 17V7L12 2Z"/>
                                    <path d="M12 22V12"/><path d="M12 12L21 7"/><path d="M12 12L3 7"/>
                                </svg>
                            </div>
                            <h2 class="welcome-title">Hola, soy Olvera AI</h2>
                            <p class="welcome-sub">&iquest;En qu&eacute; puedo ayudarte hoy?</p>
                        """),
                        class_="welcome-header"
                    ),
                    ui.div(
                        ui.div(
                            ui.HTML("""
                                <div class="suggestion-icon" style="background:#eff6ff;">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                                </div>
                                <div><div class="suggestion-title">Optimizar Código</div><div class="suggestion-sub">Refactoriza y mejora el rendimiento</div></div>
                            """),
                            class_="suggestion-card",
                            onclick="Shiny.setInputValue('suggestion_prompt', 'Ayúdame a optimizar y refactorizar mi código', {priority:'event'})"
                        ),
                        ui.div(
                            ui.HTML("""
                                <div class="suggestion-icon" style="background:#fdf4ff;">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                                </div>
                                <div><div class="suggestion-title">Generar Imágenes</div><div class="suggestion-sub">Crea arte conceptual con Olvera Image</div></div>
                            """),
                            class_="suggestion-card",
                            onclick="Shiny.setInputValue('suggestion_prompt', 'Genera una imagen artística de un paisaje futurista', {priority:'event'})"
                        ),
                        ui.div(
                            ui.HTML("""
                                <div class="suggestion-icon" style="background:#fffbeb;">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                                </div>
                                <div><div class="suggestion-title">Análisis de Datos</div><div class="suggestion-sub">Extrae insights de tus documentos</div></div>
                            """),
                            class_="suggestion-card",
                            onclick="Shiny.setInputValue('suggestion_prompt', 'Ayúdame a analizar y visualizar mis datos', {priority:'event'})"
                        ),
                        ui.div(
                            ui.HTML("""
                                <div class="suggestion-icon" style="background:#f0fdf4;">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                                </div>
                                <div><div class="suggestion-title">Brainstorming</div><div class="suggestion-sub">Genera ideas para tu próximo proyecto</div></div>
                            """),
                            class_="suggestion-card",
                            onclick="Shiny.setInputValue('suggestion_prompt', 'Necesito ideas creativas para mi próximo proyecto', {priority:'event'})"
                        ),
                        class_="suggestion-grid"
                    ),
                    class_="welcome-screen"
                )

            chat = ui.Chat(id="chat")
            chat.ui()

        # ── TOOLBAR DEL CHAT ──────────────────────────────────────────────────────────
        visible_models = providers.get_visible_models()
        default_label  = providers.get_model_label(providers.DEFAULT_MODEL)

        def split_label(full_label):
            if "(" in full_label:
                parts = full_label.rsplit("(", 1)
                return parts[0].strip(), parts[1].replace(")", "").strip()
            return full_label, ""

        default_name, default_provider = split_label(default_label)

        model_options_html = ""
        for m in visible_models:
            lbl  = providers.get_model_label(m)
            name, prov = split_label(lbl)
            name_safe = name.replace("'", "\\'")
            prov_safe = prov.replace("'", "\\'")
            model_options_html += f"""
                <div class="model-option" onclick="selectModel('{m}', '{name_safe}', '{prov_safe}')">
                    <span class="model-opt-name">{name}</span>
                    <span class="model-opt-prov">{prov}</span>
                </div>
            """

        ui.HTML(f"""
        <div class="chat-toolbar-wrapper">
            <div id="attachment-preview-area" class="attachment-preview-area"></div>
            <div class="chat-input-slot">
                <textarea id="olvera-textarea" placeholder="Escribe un mensaje..." rows="1"></textarea>
            </div>
            <div class="chat-toolbar">
                <div class="toolbar-model-area">
                    <button id="model-selector-btn" class="model-selector-btn" onclick="toggleModelDropdown()">
                        <span id="model-label">{default_name}</span>
                        <span id="model-provider" class="model-provider-tag">{default_provider}</span>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <polyline points="6 9 12 15 18 9"/>
                        </svg>
                    </button>
                    <div id="model-dropdown" class="model-dropdown">
                        {model_options_html}
                    </div>
                </div>

                <button id="web-search-btn" class="toolbar-btn" onclick="toggleWebSearch()" title="Búsqueda Web">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="2" y1="12" x2="22" y2="12"/>
                        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                    </svg>
                </button>

                <button class="toolbar-btn" onclick="triggerFileUpload()" title="Adjuntar archivo (Imagen, PDF, Word)">
                    <script>
                    function triggerFileUpload() {{
                        const input = document.getElementById('file_upload_input');
                        if (input) input.click();
                    }}
                    </script>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                    </svg>
                    <div id="file-indicator" class="file-indicator-dot"></div>
                </button>
                <div style="display:none">
                    {ui.input_file("file_upload_input", None, accept=[".png", ".jpg", ".jpeg", ".webp", ".pdf", ".docx"], multiple=False)}
                </div>

                <div style="flex:1"></div>

                <button id="floating-mic-btn" class="toolbar-btn toolbar-mic-btn" title="Dictado por voz">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
                </button>

                <button id="toolbar-send-btn" title="Enviar" onclick="sendMessage()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"/>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                    </svg>
                </button>
            </div>
        </div>
        """)



        # TAB 2: SEPHORA SENTIMIENTOS
        with ui.nav_panel("📊 Dashboard Sentimientos"):
            ui.h2("Análisis Multidimensional de Sentimientos", style="margin-top: 15px; font-weight: 800;")
            
            with ui.layout_columns(col_widths=[7, 5]):
                with ui.card():
                    ui.card_header("Evolución Temporal de Emociones")
                    @render.plot
                    def plot_sent():
                        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
                        if not df.empty:
                            df_f = sent_by_day[sent_by_day["emotion_es"].isin(input.emociones())]
                            for emo in df_f["emotion_es"].unique():
                                temp = df_f[df_f["emotion_es"] == emo]
                                color = COLOR_PALETTE.get(emo, '#000000')
                                ax.plot(temp["date"], temp["count"], label=emo, color=color, linewidth=2.5)
                        ax.set_title("Evolución de los Sentimientos por Año", color="#000000", fontsize=12, fontweight='bold')
                        ax.grid(True, color="#EAEAEA", linestyle="--")
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        fig.autofmt_xdate(rotation=35)
                        ax.legend(facecolor='#FFFFFF', edgecolor='#EAEAEA')
                        return fig

                with ui.card():
                    ui.card_header("Volumen Crítico por Emoción")
                    @render.plot
                    def plot_barras():
                        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FFFFFF')
                        if not df.empty:
                            df_f = sent_totals[sent_totals["emotion_es"].isin(input.emociones())]
                            colors = [COLOR_PALETTE.get(emo, '#000000') for emo in df_f["emotion_es"]]
                            bars = ax.bar(df_f["emotion_es"], df_f["count"], color=colors, edgecolor='#EAEAEA', width=0.6)
                        ax.set_title("Volumen Crítico por Emoción", color="#000000", fontsize=12, fontweight='bold')
                        ax.grid(True, axis='y', color="#EAEAEA", linestyle="--")
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        return fig

            with ui.layout_columns(col_widths=[6, 6]):
                with ui.card():
                    ui.card_header("Mapa de Calor Histórico")
                    @render.plot
                    def plot_heatmap():
                        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FFFFFF')
                        if not df.empty:
                            df_f = heatmap_df[heatmap_df["emotion_es"].isin(input.emociones())]
                            if not df_f.empty:
                                heat = df_f.pivot(index="emotion_es", columns="year", values="count").fillna(0)
                                sns.heatmap(heat, cmap="RdPu", ax=ax, annot=True, fmt=".0f")
                        ax.set_title("Intensidad Emocional por Año", color="#000000", fontsize=12, fontweight='bold')
                        return fig

                with ui.card():
                    ui.card_header("Calificación Promedio por Emoción")
                    @render.plot
                    def plot_rating():
                        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FFFFFF')
                        if not df.empty:
                            df_f = rating_df[rating_df["emotion_es"].isin(input.emociones())]
                            colors = [COLOR_PALETTE.get(emo, '#000000') for emo in df_f["emotion_es"]]
                            bars = ax.bar(df_f["emotion_es"], df_f["avg_rating"], color=colors, edgecolor='#EAEAEA', width=0.6)
                        ax.set_title("Calificación Promedio (Rating 1-5)", color="#000000", fontsize=12, fontweight='bold')
                        ax.set_ylim(0, 5.5)
                        ax.grid(True, axis='y', color="#EAEAEA", linestyle="--")
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        return fig

            with ui.layout_columns(col_widths=[6, 6]):
                with ui.card():
                    ui.card_header("Volumen Emocional en Top Marcas")
                    @render.plot
                    def plot_brands():
                        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
                        if not df.empty:
                            df_f = brand_df[brand_df["emotion_es"].isin(input.emociones())]
                            if not df_f.empty:
                                pivot_brand = df_f.pivot(index="brand_name", columns="emotion_es", values="count").fillna(0)
                                colors = [COLOR_PALETTE.get(col, '#000000') for col in pivot_brand.columns]
                                pivot_brand.plot(kind='barh', stacked=True, ax=ax, color=colors, edgecolor='#EAEAEA')
                        ax.set_title("Volumen en Top 10 Marcas", color="#000000", fontsize=12, fontweight='bold')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        return fig

                with ui.card():
                    ui.card_header("Perfil Emocional por Categoría")
                    @render.plot
                    def plot_categories():
                        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
                        if not df.empty:
                            df_f = cat_df[cat_df["emotion_es"].isin(input.emociones())]
                            if not df_f.empty:
                                pivot_cat = df_f.pivot(index="primary_category", columns="emotion_es", values="count").fillna(0)
                                colors = [COLOR_PALETTE.get(col, '#000000') for col in pivot_cat.columns]
                                pivot_cat.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='#EAEAEA')
                        ax.set_title("Perfil por Categoría de Producto", color="#000000", fontsize=12, fontweight='bold')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        return fig

        # TAB 3: POWER BI
        with ui.nav_panel("🏢 Dashboard Power BI"):
            ui.h2("Reporte Ejecutivo Integrado", style="margin-top: 15px; font-weight: 800;")
            with ui.card():
                ui.tags.iframe(
                    src="https://app.powerbi.com/reportEmbed?reportId=78c4a5f0-6608-4817-90ae-450c654714ac&autoAuth=true",
                    style="width: 100%; height: 750px; border: none; border-radius: 8px;"
                )

# ── HANDLERS ──────────────────────────────────────────────────────────────────

async def handle_ai_response():
    is_empty.set(False)
    messages = list(chat.messages())
    if len(messages) > 10:
        messages = messages[-10:]

    try:
        model = input.selected_model()
        if not model:
            model = providers.DEFAULT_MODEL
    except Exception:
        model = providers.DEFAULT_MODEL

    try:
        web_search = input.web_search()
        if not web_search:
            web_search = False
    except Exception:
        web_search = False

    # ── PROCESAMIENTO DE ARCHIVOS ADJUNTOS ────────────────────────────────────
    file_context = ""
    image_b64 = None
    image_url = None
    
    file_data = uploaded_file_data.get()
    if file_data:
        fname = file_data["name"].lower()
        fpath = file_data["path"]
        
        try:
            if fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
                import shutil
                import os
                # Guardar en temp_uploads para visualización persistente
                temp_name = f"up_{int(time.time())}_{fname}"
                dest_path = os.path.join("www", "temp_uploads", temp_name)
                shutil.copy(fpath, dest_path)
                image_url = f"temp_uploads/{temp_name}"
                
                with open(fpath, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
            elif fname.endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(fpath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                file_context = f"\n[CONTENIDO DE PDF: {fname}]\n{text}\n"
            elif fname.endswith(".docx"):
                import docx
                doc = docx.Document(fpath)
                text = "\n".join([p.text for p in doc.paragraphs])
                file_context = f"\n[CONTENIDO DE WORD: {fname}]\n{text}\n"
            elif fname.endswith((".txt", ".py", ".js", ".csv", ".json")):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    file_context = f"\n[CONTENIDO DE ARCHIVO: {fname}]\n{f.read()}\n"
        except Exception as e:
            print(f"Error procesando archivo: {e}")
        
        # Limpiar archivo tras procesar
        uploaded_file_data.set(None)
        await session.get_current_session().send_custom_message("file_status", {"active": False})

    # ── GENERACIÓN DE RESPUESTA ───────────────────────────────────────────────
    initial_content = ""
    if image_url:
        initial_content = f"![Adjunto]({image_url})\n\n"
    
    # Inyectar contexto de texto (PDF/Word) en el último mensaje si existe
    if file_context and messages:
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else last_msg.get('content', '')
        content = f"{content}\n\n{file_context}"
        if hasattr(last_msg, 'content'):
            last_msg.content = content
        else:
            last_msg['content'] = content

    async_gen = logic.stream_chat_response(
        messages=messages,
        model=model,
        web_search_enabled=web_search,
        image_b64=image_b64
    )
    
    # Usar un generador que incluya la imagen al principio
    async def response_with_image():
        if initial_content:
            yield initial_content
        async for chunk in async_gen:
            yield chunk

    await chat.append_message_stream(response_with_image())
    final_msgs = list(chat.messages())
    cid = current_chat_id.get()
    if cid is None:
        cid = str(int(time.time() * 1000))
        current_chat_id.set(cid)
        title = hist.auto_title(final_msgs)
        hist.add_chat(cid, title, final_msgs)
    else:
        hist.update_chat(cid, final_msgs)

@chat.on_user_submit
async def _():
    await handle_ai_response()

@reactive.effect
@reactive.event(input.new_chat)
async def _new_chat():
    current_chat_id.set(None)
    is_empty.set(True)
    await chat.clear_messages()

@reactive.effect
@reactive.event(input.load_chat_id)
async def _load_chat():
    cid = input.load_chat_id()
    if not cid: return
    data = hist.get_chat(cid)
    if not data: return
    current_chat_id.set(cid)
    is_empty.set(False)
    await chat.clear_messages()
    for m in data["messages"]:
        role    = m.role    if hasattr(m, 'role')    else m.get('role', '')
        content = m.content if hasattr(m, 'content') else m.get('content', '')
        await chat.append_message({"role": role, "content": content})

@reactive.effect
@reactive.event(input.delete_chat_id)
async def _delete_chat():
    cid = input.delete_chat_id()
    if not cid: return
    hist.delete_chat(cid)
    if current_chat_id.get() == cid:
        current_chat_id.set(None)
        is_empty.set(True)
        await chat.clear_messages()

@reactive.effect
@reactive.event(input.edit_chat_id)
def _edit_chat():
    cid = input.edit_chat_id()
    if not cid: return
    chat_data = hist.get_chat(cid)
    if not chat_data: return
    m = ui.modal(
        ui.div(
            ui.input_text("new_chat_name", "", value=chat_data["title"], width="100%"),
            ui.input_action_button("save_chat_name", "Guardar Cambios", class_="btn-new-chat"),
            style="display:flex;flex-direction:column;gap:10px;margin-top:10px;"
        ),
        title="Renombrar Chat",
        easy_close=True,
        footer=None
    )
    ui.modal_show(m)

@reactive.effect
@reactive.event(input.save_chat_name)
def _save_chat_name():
    new_name = input.new_chat_name()
    cid = input.edit_chat_id()
    if new_name and cid:
        hist.update_chat_title(cid, new_name)
        ui.modal_remove()

@reactive.effect
@reactive.event(input.suggestion_prompt)
async def _suggestion():
    prompt = input.suggestion_prompt()
    if prompt:
        await chat.append_message({"role": "user", "content": prompt})
        await handle_ai_response()

# ✅ DESPUÉS
@reactive.effect
@reactive.event(input.voice_audio_payload)
async def _voice_input():
    b64_data = input.voice_audio_payload()
    if not b64_data: return
    text = None
    try:
        text = await logic.transcribe_audio(b64_data)
    except Exception as e:
        print(f"Error transcribiendo audio: {e}")
    finally:
        s = session.get_current_session()
        if s:
            await s.send_custom_message("transcription_done", {"status": "ok"})
    if text and text.strip():
        await chat.append_message({"role": "user", "content": text.strip()})
        await handle_ai_response()

@reactive.effect
@reactive.event(input.file_upload_input)
async def _handle_upload():
    f = input.file_upload_input()
    if f:
        path = f[0]["datapath"]
        name = f[0]["name"]
        uploaded_file_data.set({"name": name, "path": path})
        
        preview_data = None
        is_img = name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        if is_img:
            with open(path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
                preview_data = f"data:image/jpeg;base64,{b64}"
        
        s = session.get_current_session()
        if s:
            await s.send_custom_message("file_status", {
                "active": True, 
                "preview": preview_data, 
                "type": "image" if is_img else "file"
            })

@reactive.effect
@reactive.event(input.file_pasted)
async def _handle_paste():
    p = input.file_pasted()
    if p:
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".png")
        img_data = base64.b64decode(p["data"])
        with os.fdopen(fd, 'wb') as f:
            f.write(img_data)
        
        uploaded_file_data.set({"name": p["name"], "path": path})
        s = session.get_current_session()
        if s:
            await s.send_custom_message("file_status", {
                "active": True, 
                "preview": f"data:image/png;base64,{p['data']}", 
                "type": "image"
            })

@reactive.effect
@reactive.event(input.remove_attachment)
async def _remove_attachment():
    uploaded_file_data.set(None)
    s = session.get_current_session()
    if s:
        await s.send_custom_message("file_status", {"active": False})

ui.head_content(ui.HTML("""
<style>
.attachment-preview-area {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 0 12px 8px 12px;
}
.attachment-preview-area:empty { display: none; }
.attachment-chip {
    position: relative;
    width: 56px;
    height: 56px;
    border-radius: 10px;
    background: #1f2937;
    border: 1px solid rgba(255,255,255,0.1);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.attachment-chip img { width: 100%; height: 100%; object-fit: cover; }
.attachment-remove-btn {
    position: absolute;
    top: -4px;
    right: -4px;
    width: 18px;
    height: 18px;
    background: #ef4444;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid #111827;
    cursor: pointer;
    font-size: 12px;
}
.file-indicator-dot {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    border: 2px solid #111827;
    display: none;
}
.toolbar-btn { position: relative; }
/* Limitar tamaño de imágenes en el chat - Estilo miniatura premium */
.shiny-chat-message img {
    max-width: 350px;
    max-height: 250px;
    width: auto;
    height: auto;
    object-fit: contain;
    border-radius: 12px;
    margin: 12px 0;
    cursor: zoom-in;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(0,0,0,0.2);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.shiny-chat-message img:hover {
    transform: scale(1.02);
}
</style>
"""))

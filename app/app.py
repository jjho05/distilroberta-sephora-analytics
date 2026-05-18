import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
import time
from shiny import App, render, ui, reactive

import logic
import providers


# ==========================================
# 1. OPTIMIZACIÓN Y CARGA DE DATOS (PRO SUITE)
# ==========================================

# Definimos la ruta local del dataset completo (36MB) para máximo contexto analítico
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "sephora_with_emotions.csv")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"No se encontró el archivo de datos en: {DATA_FILE}")

print("📥 Cargando base de datos completa de Sephora (36 MB)...")
df = pd.read_csv(DATA_FILE)

# Convertir fechas de forma segura y veloz
df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")
df["year"] = df["submission_time"].dt.year

# Mapeo de emociones al español para la visualización ejecutiva
diccionario_es = {
    'anger': 'Ira',
    'disgust': 'Desagrado', 
    'fear': 'Miedo',
    'joy': 'Alegría',
    'neutral': 'Neutral',
    'sadness': 'Tristeza',
    'surprise': 'Sorpresa'
}
df["emotion_es"] = df["emotion"].map(diccionario_es).fillna('Neutral')

# ----------------------------------------------------
# PRE-CÁLCULOS CLAVE (Máxima Velocidad Reactiva)
# ----------------------------------------------------

# 1. Evolución temporal de emociones
sent_by_day = df.groupby([df["submission_time"].dt.to_period("M"), "emotion_es"]).size().reset_index(name="count")
sent_by_day["date"] = sent_by_day["submission_time"].dt.to_timestamp()

# 2. Totales por emoción
sent_totals = df.groupby("emotion_es").size().reset_index(name="count")

# 3. Mapa de calor (Año vs Emoción)
heatmap_df = df.groupby(["year", "emotion_es"]).size().reset_index(name="count")

# 4. Rating promedio por emoción (Métrica de validación de NLP)
rating_df = df.groupby("emotion_es")["rating"].mean().reset_index(name="avg_rating")

# 5. Top 10 marcas más comentadas y sus emociones
brand_totals = df.groupby(["brand_name", "emotion_es"]).size().reset_index(name="count")
top_brands = df["brand_name"].value_counts().nlargest(10).index
brand_df = brand_totals[brand_totals["brand_name"].isin(top_brands)]

# 6. Emociones distribuidas por categoría de producto
cat_df = df.groupby(["primary_category", "emotion_es"]).size().reset_index(name="count")


# Paleta de colores oficial "Olvera BI Light Premium"
COLOR_PALETTE = {
    'Alegría': '#D4AF37',    # Dorado Premium
    'Desagrado': '#FF5733',  # Coral
    'Ira': '#FF0055',        # Rosa Neón/Rojo Sephora
    'Miedo': '#8E44AD',      # Púrpura oscuro
    'Neutral': '#7F8C8D',    # Gris
    'Tristeza': '#3498DB',   # Azul suave
    'Sorpresa': '#2ECC71'    # Verde Neón
}

# ==========================================
# 2. INTERFAZ DE USUARIO (UX/UI PREMIUM - MODO CLARO)
# ==========================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

body {
    background-color: #F8F9FA !important;
    color: #1A1A1A !important;
    font-family: 'Montserrat', 'Segoe UI', Roboto, sans-serif;
}
.sidebar {
    background-color: #FFFFFF !important;
    border-right: 1px solid #EAEAEA !important;
}
.card {
    background: #FFFFFF !important;
    border: 1px solid #EAEAEA !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 25px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    overflow: hidden;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08) !important;
}
.card-header {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #F0F0F2 !important;
    color: #000000 !important;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    padding: 15px 20px !important;
}
h2, h3, h4 {
    font-family: 'Montserrat', sans-serif !important;
    color: #000000 !important;
    font-weight: 800;
}
.nav-tabs .nav-link.active {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border-color: #000000 !important;
    border-radius: 6px;
    font-weight: 700;
}
.nav-tabs .nav-link {
    color: #666666 !important;
    font-weight: 600;
    border: none !important;
    padding: 10px 20px !important;
}
.form-check-input:checked {
    background-color: #FF0055 !important;
    border-color: #FF0055 !important;
}
.sephora-stripes {
    height: 8px;
    background: repeating-linear-gradient(90deg, #000000, #000000 15px, #FFFFFF 15px, #FFFFFF 30px);
    width: 100%;
}
.auth-warning {
    background: rgba(255, 0, 85, 0.02);
    border: 1px solid rgba(255, 0, 85, 0.1);
    border-left: 4px solid #FF0055;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 25px;
}
.explanation-box {
    background: #F8F9FA;
    border-top: 1px solid #EAEAEA;
    padding: 15px 20px;
    border-radius: 0 0 12px 12px;
    font-size: 0.88rem;
    color: #444444;
    line-height: 1.6;
}
.explanation-title {
    font-weight: 700;
    color: #FF0055;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}
/* Estilos del chat para garantizar altura perfecta y evitar solapamientos */
.shiny-chat {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 0 !important;
}
.shiny-chat-messages {
    flex-grow: 1 !important;
    overflow-y: auto !important;
    padding-bottom: 140px !important; /* Espacio extra para que ningún texto pase detrás de la barra de input */
}
"""

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.HTML("""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #EAEAEA;">
                <div style="width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(135deg, #10A37F, #0E8E6D); display: flex; align-items: center; justify-content: center; color: white; font-size: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                    ✨
                </div>
                <div>
                    <div style="font-weight: 800; font-size: 1.2rem; color: #1a1a1a; font-family: 'Montserrat', sans-serif; letter-spacing: 0.5px;">Olvera BI</div>
                    <div style="font-size: 0.75rem; color: #666666; font-weight: 500; letter-spacing: 0.5px;">Programación para Ciencia de Datos</div>
                </div>
            </div>
            """)
        ),
        ui.input_checkbox_group(
            "emociones",
            ui.span("FILTRAR EMOCIONES:", style="font-weight: 700; color: #000000; font-size: 0.85rem; letter-spacing: 1px;"),
            choices=sorted(list(diccionario_es.values())),
            selected=sorted(list(diccionario_es.values())),
        ),
        ui.hr(style="border-color: #EAEAEA; margin-top: 25px; margin-bottom: 25px;"),
        ui.div(
            ui.p("💡 INFORMACIÓN:", style="font-weight: bold; color: #000000; font-size: 0.85em; letter-spacing: 0.5px;"),
            ui.p("Este panel interactivo conecta la base de datos con los resultados del modelo de deep learning DistilRoBERTa.", style="font-size: 0.82em; color: #555555; line-height: 1.5;"),
            style="background: #F8F9FA; padding: 15px; border-radius: 8px; border: 1px solid #EAEAEA;"
        ),
        width=290,
        style="padding: 25px;"
    ),

    ui.div(
        ui.div(style="height: 6px; background: repeating-linear-gradient(90deg, #000, #000 15px, #fff 15px, #fff 30px); width: 100%; border-radius: 4px 4px 0 0;"), # Stripes
    ),

    ui.navset_tab(
        # TAB 1: Análisis Nativo de IA
        ui.nav_panel(
            "Dashboard Sentimientos",
            ui.div(
                ui.h2("Análisis Multidimensional de Sentimientos (Skincare)", style="margin-top: 15px; margin-bottom: 25px; font-weight: 800; letter-spacing: -0.5px;"),
                
                # GRID FILA 1
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Evolución Temporal de Emociones (Skincare)"),
                        ui.output_plot("plot_sent"),
                        ui.div(
                            ui.div("📈 ANÁLISIS DE TENDENCIA TEMPORAL", class_="explanation-title"),
                            ui.p("Rastrea cómo las opiniones y emociones de los clientes han cambiado a lo largo del tiempo. Esencial para evaluar si los picos emocionales corresponden a campañas publicitarias o rediseños de fórmulas de productos."),
                            class_="explanation-box"
                        )
                    ),
                    ui.card(
                        ui.card_header("Volumen Crítico por Emoción"),
                        ui.output_plot("plot_barras"),
                        ui.div(
                            ui.div("📊 ANÁLISIS DE MASA CRÍTICA", class_="explanation-title"),
                            ui.p("Muestra la cantidad total de reseñas clasificadas en cada emoción. La fuerte dominancia de 'Alegría' resalta la satisfacción general, mientras que picos en 'Ira' o 'Desagrado' sirven como alertas inmediatas de calidad."),
                            class_="explanation-box"
                        )
                    ),
                    col_widths=[7, 5]
                ),
                
                # GRID FILA 2
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Mapa de Calor Histórico (Emociones vs Años)"),
                        ui.output_plot("plot_heatmap"),
                        ui.div(
                            ui.div("🔥 ANÁLISIS DE DENSIDAD HISTÓRICA", class_="explanation-title"),
                            ui.p("Representa visualmente la concentración de reseñas por año. Permite identificar anomalías específicas o ver el crecimiento masivo en la retroalimentación de los usuarios a partir de 2018."),
                            class_="explanation-box"
                        )
                    ),
                    ui.card(
                        ui.card_header("Calificación Promedio por Emoción"),
                        ui.output_plot("plot_rating"),
                        ui.div(
                            ui.div("⭐ CORRELACIÓN DE SATISFACCIÓN (RATING)", class_="explanation-title"),
                            ui.p("Calcula la calificación promedio (estrellas) que los clientes otorgaron en cada emoción. Valida el motor de IA de DistilRoBERTa: las emociones negativas obtienen promedios bajos, mientras que 'Alegría' roza la excelencia."),
                            class_="explanation-box"
                        )
                    ),
                    col_widths=[6, 6]
                ),

                # GRID FILA 3
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Volumen Emocional en Top 10 Marcas"),
                        ui.output_plot("plot_brands"),
                        ui.div(
                            ui.div("🏆 COMPORTAMIENTO DE MARCAS LÍDERES", class_="explanation-title"),
                            ui.p("Compara la proporción de sentimientos entre las 10 marcas de Skincare con mayor cantidad de menciones. Ideal para auditoría competitiva e identificar líderes de satisfacción."),
                            class_="explanation-box"
                        )
                    ),
                    ui.card(
                        ui.card_header("Perfil Emocional por Categoría de Producto"),
                        ui.output_plot("plot_categories"),
                        ui.div(
                            ui.div("🛍️ IMPACTO EMOCIONAL POR CATEGORÍA", class_="explanation-title"),
                            ui.p("Distribuye las emociones según la línea principal de producto (Skincare, Haircare, etc.). Revela si los clientes son sustancialmente más exigentes con ciertos tratamientos de cuidado personal."),
                            class_="explanation-box"
                        )
                    ),
                    col_widths=[6, 6]
                ),
                style="padding: 10px;"
            )
        ),

        # TAB 2: Integración de Power BI
        ui.nav_panel(
            "💬 Olvera BI Copilot",
            ui.div(
                ui.card(
                    ui.output_ui("welcome_ui"),
                    ui.chat_ui("chat"),
                    style="height: 720px; border: 1px solid #EAEAEA; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden;"
                ),
                ui.output_ui("chat_toolbar_ui"),
                style="padding: 10px;"
            )
        ),
        ui.nav_panel(
            "Dashboard Power BI",
            ui.div(
                ui.h2("Reporte Ejecutivo Integrado", style="margin-top: 15px; margin-bottom: 20px; font-weight: 800;"),
                ui.div(
                    ui.h4("🔑 ¿Por qué se muestra el botón de inicio de sesión?", style="color: #FF0055; font-weight: bold; margin-bottom: 8px;"),
                    ui.p("Por seguridad corporativa de Microsoft, el inicio de sesión automático está bloqueado dentro de marcos incrustados en sitios de terceros como Hugging Face.", style="color: #444444; font-size: 0.9em; margin-bottom: 12px;"),
                    ui.div(
                        ui.a(
                            "👉 Iniciar Sesión / Abrir en Pestaña Nueva", 
                            href="https://app.powerbi.com/reportEmbed?reportId=78c4a5f0-6608-4817-90ae-450c654714ac&autoAuth=true", 
                            target="_blank",
                            class_="btn btn-dark",
                            style="background-color: #000000; border-color: #000000; padding: 8px 18px; font-weight: bold; font-size: 0.9em; border-radius: 6px; color: white; text-decoration: none; display: inline-block; margin-right: 15px;"
                        ),
                        ui.span("Recomendado: Genera el link público en tu Power BI ('Publicar en la web') para cargarlo directamente sin inicios de sesión.", style="color: #666666; font-size: 0.85em;"),
                        style="margin-top: 10px;"
                    ),
                    class_="auth-warning"
                ),
                ui.card(
                    ui.tags.iframe(
                        src="https://app.powerbi.com/reportEmbed?reportId=78c4a5f0-6608-4817-90ae-450c654714ac&autoAuth=true",
                        style="width: 100%; height: 750px; border: none; border-radius: 8px;"
                    ),
                    style="padding: 5px;"
                ),
                style="padding: 10px;"
            )
        )
    ),
    ui.tags.head(
        ui.tags.style(CUSTOM_CSS)
    ),
    title=None
)

# ==========================================
# 3. LÓGICA DEL SERVIDOR (FAST REACTIVE)
# ==========================================

def server(input, output, session):

    is_empty = reactive.Value(True)
    chat = ui.Chat(id="chat")

    @chat.on_user_submit
    async def _():
        is_empty.set(False)
        messages = list(chat.messages())
        if len(messages) > 10:
            messages = messages[-10:]
            
        # Inyectar el contexto completo del dataset al último mensaje
        try:
            emos = input.emociones()
            df_totals = sent_totals[sent_totals["emotion_es"].isin(emos)]
            df_brands = brand_df[brand_df["emotion_es"].isin(emos)]
            df_cats = cat_df[cat_df["emotion_es"].isin(emos)]
            df_ratings = rating_df[rating_df["emotion_es"].isin(emos)]
            
            # Obtener una muestra aleatoria (o los top) de reviews crudos (hasta 5)
            df_raw = df[df["emotion_es"].isin(emos)]
            sample_reviews = "Sin reseñas disponibles."
            if not df_raw.empty:
                # Tomar los 5 más recientes o aleatorios que tengan texto
                df_raw_sample = df_raw.dropna(subset=['review_text']).head(5)
                sample_reviews = "\n".join([f"- [{row['brand_name']}] Rating: {row['rating']}⭐: \"{row['review_text']}\"" for _, row in df_raw_sample.iterrows()])
            
            ctx = f"\n\n[CONTEXTO OLVERA BI - FILTROS ACTIVOS: {', '.join(emos)}]\n"
            ctx += "-- TOTALES POR EMOCIÓN --\n" + (df_totals.to_string(index=False) if not df_totals.empty else "N/A") + "\n\n"
            ctx += "-- COMPORTAMIENTO TOP 10 MARCAS --\n" + (df_brands.to_string(index=False) if not df_brands.empty else "N/A") + "\n\n"
            ctx += "-- EMOCIONES POR CATEGORÍA DE PRODUCTO --\n" + (df_cats.to_string(index=False) if not df_cats.empty else "N/A") + "\n\n"
            ctx += "-- RATING PROMEDIO (1-5 ESTRELLAS) VS EMOCIÓN --\n" + (df_ratings.to_string(index=False) if not df_ratings.empty else "N/A") + "\n\n"
            ctx += "-- MUESTRA DE RESEÑAS REALES DE USUARIOS --\n" + sample_reviews
            
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                last_msg.content += ctx
            elif isinstance(last_msg, dict) and 'content' in last_msg:
                last_msg['content'] += ctx
        except Exception as e:
            print(f"Error inyectando contexto masivo: {e}")

        async_gen = logic.stream_chat_response(
            messages=messages,
            model=providers.DEFAULT_MODEL,
            web_search_enabled=False,
            image_b64=None
        )
        await chat.append_message_stream(async_gen)

    @reactive.Effect
    @reactive.event(input.suggestion_click)
    async def _handle_suggestion():
        prompt = input.suggestion_click()
        if not prompt:
            return
        is_empty.set(False)
        await chat.append_message({"role": "user", "content": prompt})
        # Also trigger the AI response immediately
        try:
            emos = input.emociones()
            df_totals = sent_totals[sent_totals["emotion_es"].isin(emos)]
            df_brands = brand_df[brand_df["emotion_es"].isin(emos)]
            df_cats = cat_df[cat_df["emotion_es"].isin(emos)]
            df_ratings = rating_df[rating_df["emotion_es"].isin(emos)]
            
            # Obtener una muestra aleatoria (o los top) de reviews crudos (hasta 5)
            df_raw = df[df["emotion_es"].isin(emos)]
            sample_reviews = "Sin reseñas disponibles."
            if not df_raw.empty:
                # Tomar los 5 más recientes o aleatorios que tengan texto
                df_raw_sample = df_raw.dropna(subset=['review_text']).head(5)
                sample_reviews = "\n".join([f"- [{row['brand_name']}] Rating: {row['rating']}⭐: \"{row['review_text']}\"" for _, row in df_raw_sample.iterrows()])
            
            ctx = f"\n\n[CONTEXTO OLVERA BI - FILTROS ACTIVOS: {', '.join(emos)}]\n"
            ctx += "-- TOTALES POR EMOCIÓN --\n" + (df_totals.to_string(index=False) if not df_totals.empty else "N/A") + "\n\n"
            ctx += "-- COMPORTAMIENTO TOP 10 MARCAS --\n" + (df_brands.to_string(index=False) if not df_brands.empty else "N/A") + "\n\n"
            ctx += "-- EMOCIONES POR CATEGORÍA DE PRODUCTO --\n" + (df_cats.to_string(index=False) if not df_cats.empty else "N/A") + "\n\n"
            ctx += "-- RATING PROMEDIO (1-5 ESTRELLAS) VS EMOCIÓN --\n" + (df_ratings.to_string(index=False) if not df_ratings.empty else "N/A") + "\n\n"
            ctx += "-- MUESTRA DE RESEÑAS REALES DE USUARIOS --\n" + sample_reviews
            
            enriched_prompt = prompt + ctx
        except Exception as e:
            print(f"Error inyectando contexto en sugerencia: {e}")
            enriched_prompt = prompt
        
        # Obtener el historial completo y actualizar el último mensaje
        msgs = chat.messages(format="dict")
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = enriched_prompt
        else:
            msgs.append({"role": "user", "content": enriched_prompt})
            
        msgs.insert(0, {
            "role": "system", 
            "content": "Eres Olvera BI Copilot, un analista de datos avanzado y experto en Power BI..."
        })
        
        async_gen = logic.stream_chat_response(
            messages=msgs,
            model=providers.DEFAULT_MODEL,
            web_search_enabled=False,
            image_b64=None
        )
        await chat.append_message_stream(async_gen)

    @output
    @render.ui
    def welcome_ui():
        if not is_empty.get():
            return ui.div()
        return ui.div(
            ui.div(
                ui.HTML("""
                    <div class="welcome-icon" style="width:64px;height:64px;background:#000;border-radius:16px;display:flex;align-items:center;justify-content:center;margin:0 auto 16px;box-shadow:0 10px 30px rgba(0,0,0,0.15)">
                        <svg viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.5' style='width:36px;height:36px;'>
                            <path d='M12 2L3 7V17L12 22L21 17V7L12 2Z'/>
                            <path d='M12 22V12'/><path d='M12 12L21 7'/><path d='M12 12L3 7'/>
                        </svg>
                    </div>
                    <h2 style='font-family:Montserrat,sans-serif;font-weight:800;font-size:1.4rem;color:#000;margin-bottom:8px;'>Hola, soy Olvera BI</h2>
                    <p style='color:#666;font-size:0.95rem;'>Analizo en tiempo real los datos filtrados del dashboard. ¿En qué te puedo ayudar?</p>
                """),
                style="text-align:center;padding:30px 20px 20px;"
            ),
            ui.div(
                ui.div(
                    ui.HTML("<div style='font-weight:700;color:#000;font-size:0.9rem;margin-bottom:4px;'>📊 Analizar emociones</div><div style='color:#666;font-size:0.82rem;'>Explica los datos actuales del dashboard</div>"),
                    onclick="Shiny.setInputValue('suggestion_click', 'Analiza y explica las emociones del dataset de Sephora que están activas ahora', {priority:'event'})",
                    style="background:#F8F9FA;border:1px solid #EAEAEA;border-radius:10px;padding:14px 16px;cursor:pointer;transition:all 0.2s;"
                ),
                ui.div(
                    ui.HTML("<div style='font-weight:700;color:#000;font-size:0.9rem;margin-bottom:4px;'>🎯 Insights de marcas</div><div style='color:#666;font-size:0.82rem;'>Identifica tendencias en top 10 marcas</div>"),
                    onclick="Shiny.setInputValue('suggestion_click', '¿Qué marcas tienen mejor perfil emocional y cuáles necesitan atención?', {priority:'event'})",
                    style="background:#F8F9FA;border:1px solid #EAEAEA;border-radius:10px;padding:14px 16px;cursor:pointer;transition:all 0.2s;"
                ),
                ui.div(
                    ui.HTML("<div style='font-weight:700;color:#000;font-size:0.9rem;margin-bottom:4px;'>⚠️ Alertas de calidad</div><div style='color:#666;font-size:0.82rem;'>Detecta picos de Ira y Desagrado</div>"),
                    onclick="Shiny.setInputValue('suggestion_click', 'Identifica alertas de calidad según los picos de Ira y Desagrado en el dataset', {priority:'event'})",
                    style="background:#F8F9FA;border:1px solid #EAEAEA;border-radius:10px;padding:14px 16px;cursor:pointer;transition:all 0.2s;"
                ),
                ui.div(
                    ui.HTML("<div style='font-weight:700;color:#000;font-size:0.9rem;margin-bottom:4px;'>⭐ Rating vs emoción</div><div style='color:#666;font-size:0.82rem;'>Correlaciona calificación y sentimiento</div>"),
                    onclick="Shiny.setInputValue('suggestion_click', 'Explica la correlación entre el rating promedio y cada emoción detectada', {priority:'event'})",
                    style="background:#F8F9FA;border:1px solid #EAEAEA;border-radius:10px;padding:14px 16px;cursor:pointer;transition:all 0.2s;"
                ),
                style="display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 20px 20px;"
            ),
            style="max-width:680px;margin:0 auto;"
        )

    @output
    @render.ui
    def chat_toolbar_ui():
        return ui.HTML("""
            <div style='border-top:1px solid #EAEAEA;padding:12px 10px 0;margin-top:8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
                <span style='font-size:0.78rem;color:#999;font-weight:600;letter-spacing:0.5px;'>OLVERA BI COPILOT</span>
                <span style='font-size:0.78rem;color:#ccc;'>|</span>
                <span style='font-size:0.78rem;color:#666;'>Gemini 3 Flash &bull; Contexto del dashboard inyectado automáticamente</span>
            </div>
        """)

    # Filtro reactivo ultra-veloz

    @reactive.Calc
    def filtered_time_data():
        emos = input.emociones()
        return sent_by_day[sent_by_day["emotion_es"].isin(emos)]

    @reactive.Calc
    def filtered_totals():
        emos = input.emociones()
        return sent_totals[sent_totals["emotion_es"].isin(emos)]

    @reactive.Calc
    def filtered_heatmap():
        emos = input.emociones()
        return heatmap_df[heatmap_df["emotion_es"].isin(emos)]

    # ---- GRÁFICA 1: Evolución Temporal ----
    @output
    @render.plot
    def plot_sent():
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        df_f = filtered_time_data()
        
        for emo in df_f["emotion_es"].unique():
            temp = df_f[df_f["emotion_es"] == emo]
            color = COLOR_PALETTE.get(emo, '#000000')
            ax.plot(temp["date"], temp["count"], label=emo, color=color, linewidth=2.5)
            
        ax.set_title("Evolución de los Sentimientos por Año", color="#000000", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Línea de Tiempo", color="#555555", fontsize=10)
        ax.set_ylabel("Cantidad de Reseñas", color="#555555", fontsize=10)
        
        # Ajustar rejilla y ejes para el Light Theme
        ax.grid(True, color="#EAEAEA", linestyle="--")
        ax.spines['bottom'].set_color('#EAEAEA')
        ax.spines['left'].set_color('#EAEAEA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#444444')
        
        fig.autofmt_xdate(rotation=35)
        
        # Leyenda Premium
        leg = ax.legend(facecolor='#FFFFFF', edgecolor='#EAEAEA')
        for text in leg.get_texts():
            text.set_color('#000000')
            
        plt.tight_layout()
        return fig

    # ---- GRÁFICA 2: Volumen de Emociones ----
    @output
    @render.plot
    def plot_barras():
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        df_f = filtered_totals()
        
        if not df_f.empty:
            colors = [COLOR_PALETTE.get(emo, '#000000') for emo in df_f["emotion_es"]]
            bars = ax.bar(df_f["emotion_es"], df_f["count"], color=colors, edgecolor='#EAEAEA', width=0.6)
            
            # Agregar etiquetas de valor en las barras
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 puntos de desfase vertical
                            textcoords="offset points",
                            ha='center', va='bottom', color='#000000', fontsize=9, fontweight='bold')
                            
        ax.set_title("Volumen Crítico por Emoción", color="#000000", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Emoción Clasificada", color="#555555")
        ax.set_ylabel("Total Reseñas", color="#555555")
        
        ax.grid(True, axis='y', color="#EAEAEA", linestyle="--")
        ax.spines['bottom'].set_color('#EAEAEA')
        ax.spines['left'].set_color('#EAEAEA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#444444')
        
        plt.tight_layout()
        return fig

    # ---- GRÁFICA 3: Mapa de Calor ----
    @output
    @render.plot
    def plot_heatmap():
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        df_f = filtered_heatmap()
        
        if df_f.empty:
            ax.text(0.5, 0.5, "Selecciona al menos una emoción en el panel lateral.", 
                    horizontalalignment='center', verticalalignment='center', color='#000000', transform=ax.transAxes)
        else:
            # Pivotar datos para generar la matriz del Heatmap
            heat = df_f.pivot(index="emotion_es", columns="year", values="count").fillna(0)
            
            # Dibujar mapa de calor premium en tonos rosa/purpura
            sns.heatmap(heat, cmap="RdPu", ax=ax, annot=True, fmt=".0f", 
                        cbar_kws={'label': 'Conteo de Reseñas'}, annot_kws={"size": 10, "weight": "bold"})
            
            # Estilizar barra de colores
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='#000000')
            cbar.ax.tick_params(labelsize=10, labelcolor='#000000')
            cbar.set_label('Conteo de Reseñas', color='#000000', size=10)
            
            ax.set_title("Intensidad Emocional por Año", color="#000000", fontsize=12, fontweight='bold', pad=15)
            ax.set_ylabel("Emoción", color="#555555")
            ax.set_xlabel("Año de la Reseña", color="#555555")
            
            ax.tick_params(colors='#000000')
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        return fig

    # ---- GRÁFICA 4: Calificación Promedio por Emoción ----
    @output
    @render.plot
    def plot_rating():
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        emos = input.emociones()
        df_f = rating_df[rating_df["emotion_es"].isin(emos)]
        
        if not df_f.empty:
            colors = [COLOR_PALETTE.get(emo, '#000000') for emo in df_f["emotion_es"]]
            bars = ax.bar(df_f["emotion_es"], df_f["avg_rating"], color=colors, edgecolor='#EAEAEA', width=0.6)
            
            # Agregar etiquetas con el rating promedio
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f} ★',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 puntos de desfase vertical
                            textcoords="offset points",
                            ha='center', va='bottom', color='#000000', fontsize=9, fontweight='bold')
                            
        ax.set_title("Calificación Promedio por Emoción (Rating 1-5)", color="#000000", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Emoción", color="#555555")
        ax.set_ylabel("Rating Promedio", color="#555555")
        ax.set_ylim(0, 5.5)
        
        ax.grid(True, axis='y', color="#EAEAEA", linestyle="--")
        ax.spines['bottom'].set_color('#EAEAEA')
        ax.spines['left'].set_color('#EAEAEA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#444444')
        
        plt.tight_layout()
        return fig

    # ---- GRÁFICA 5: Volumen Emocional en Top 10 Marcas ----
    @output
    @render.plot
    def plot_brands():
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        emos = input.emociones()
        df_f = brand_df[brand_df["emotion_es"].isin(emos)]
        
        if not df_f.empty:
            # Pivotar para generar barras apiladas
            pivot_brand = df_f.pivot(index="brand_name", columns="emotion_es", values="count").fillna(0)
            
            # Generar barras apiladas
            colors = [COLOR_PALETTE.get(col, '#000000') for col in pivot_brand.columns]
            pivot_brand.plot(kind='barh', stacked=True, ax=ax, color=colors, edgecolor='#EAEAEA')
            
        ax.set_title("Volumen Emocional en Top 10 Marcas", color="#000000", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Número de Reseñas", color="#555555")
        ax.set_ylabel("Marca", color="#555555")
        
        ax.grid(True, axis='x', color="#EAEAEA", linestyle="--")
        ax.spines['bottom'].set_color('#EAEAEA')
        ax.spines['left'].set_color('#EAEAEA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#444444')
        
        leg = ax.legend(facecolor='#FFFFFF', edgecolor='#EAEAEA', title="Emoción")
        for text in leg.get_texts():
            text.set_color('#000000')
            
        plt.tight_layout()
        return fig

    # ---- GRÁFICA 6: Perfil Emocional por Categoría de Producto ----
    @output
    @render.plot
    def plot_categories():
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        emos = input.emociones()
        df_f = cat_df[cat_df["emotion_es"].isin(emos)]
        
        if not df_f.empty:
            # Pivotar para generar barras apiladas
            pivot_cat = df_f.pivot(index="primary_category", columns="emotion_es", values="count").fillna(0)
            
            # Generar barras apiladas
            colors = [COLOR_PALETTE.get(col, '#000000') for col in pivot_cat.columns]
            pivot_cat.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='#EAEAEA')
            
        ax.set_title("Perfil Emocional por Categoría de Producto", color="#000000", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Categoría", color="#555555")
        ax.set_ylabel("Total Reseñas", color="#555555")
        
        ax.grid(True, axis='y', color="#EAEAEA", linestyle="--")
        ax.spines['bottom'].set_color('#EAEAEA')
        ax.spines['left'].set_color('#EAEAEA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#444444')
        plt.xticks(rotation=15)
        
        leg = ax.legend(facecolor='#FFFFFF', edgecolor='#EAEAEA', title="Emoción")
        for text in leg.get_texts():
            text.set_color('#000000')
            
        plt.tight_layout()
        return fig

# ==========================================
# 4. INSTANCIACIÓN DE APLICACIÓN
# ==========================================
app = App(app_ui, server)
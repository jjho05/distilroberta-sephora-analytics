import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from shiny import App, render, ui, reactive

# ==========================================
# 1. OPTIMIZACIÓN Y CARGA DE DATOS (PRO SUITE)
# ==========================================

# Definimos la ruta local del dataset optimizado
DATA_FILE = os.path.join(os.path.dirname(__file__), "reviews_emociones_opt.csv")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"No se encontró el archivo de datos en: {DATA_FILE}")

print("📥 Cargando base de datos optimizada (5.7 MB)...")
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

# Agrupaciones pre-calculadas para máxima velocidad de respuesta (Reactive Performance)
sent_by_day = df.groupby([df["submission_time"].dt.to_period("M"), "emotion_es"]).size().reset_index(name="count")
sent_by_day["date"] = sent_by_day["submission_time"].dt.to_timestamp()

sent_totals = df.groupby("emotion_es").size().reset_index(name="count")

heatmap_df = df.groupby(["year", "emotion_es"]).size().reset_index(name="count")

# Paleta de colores oficial "Sephora Dark Premium"
COLOR_PALETTE = {
    'Alegría': '#D4AF37',    # Dorado Premium
    'Desagrado': '#FF5733',  # Coral
    'Ira': '#FF0055',        # Rosa Neón/Rojo
    'Miedo': '#8E44AD',      # Púrpura oscuro
    'Neutral': '#7F8C8D',    # Gris
    'Tristeza': '#3498DB',   # Azul suave
    'Sorpresa': '#2ECC71'    # Verde Neón
}

# ==========================================
# 2. INTERFAZ DE USUARIO (UX/UI PREMIUM)
# ==========================================

# Estilos CSS personalizados para lograr el efecto Glassmorphism y Dark Theme
CUSTOM_CSS = """
body {
    background-color: #0E0E10 !important;
    color: #ECECF1 !important;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.sidebar {
    background-color: #16161A !important;
    border-right: 1px solid #2A2A35 !important;
}
.card {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    margin-bottom: 20px;
}
.card-header {
    background-color: rgba(0, 0, 0, 0.2) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #FFFFFF !important;
    font-weight: 600;
}
h2, h3, h4 {
    color: #FFFFFF !important;
    font-weight: 700;
}
.nav-tabs .nav-link.active {
    background-color: #FF0055 !important;
    color: white !important;
    border-color: #FF0055 !important;
    border-radius: 6px;
}
.nav-tabs .nav-link {
    color: #A0A0B0 !important;
}
.form-check-input:checked {
    background-color: #FF0055 !important;
    border-color: #FF0055 !important;
}
"""

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h3("Sephora AI Suite", style="color: #FF0055; margin-bottom: 5px;"),
            ui.p("Sentiment Control Panel", style="color: #8E8E93; font-size: 0.9em; margin-bottom: 25px;"),
        ),
        ui.input_checkbox_group(
            "emociones",
            ui.span("Filtrar Emociones:", style="font-weight: 600; color: #FFFFFF;"),
            choices=sorted(list(diccionario_es.values())),
            selected=sorted(list(diccionario_es.values())),
        ),
        ui.hr(style="border-color: #2A2A35;"),
        ui.div(
            ui.p("💡 Consejo:", style="font-weight: bold; color: #FF0055;"),
            ui.p("Este panel interactúa dinámicamente con los algoritmos de IA de DistilRoBERTa.", style="font-size: 0.85em; color: #AEAEB2;"),
            style="margin-top: 20px;"
        ),
        width=300,
        style="padding: 20px;"
    ),

    ui.navset_tab(
        # TAB 1: Análisis Nativo de IA
        ui.nav_panel(
            "Dashboard Sentimientos",
            ui.div(
                ui.h2("Análisis Multidimensional de Sentimientos", style="margin-top: 15px; margin-bottom: 20px;"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Evolución Temporal de Emociones (Skincare)"),
                        ui.output_plot("plot_sent"),
                    ),
                    ui.card(
                        ui.card_header("Volumen Crítico por Emoción"),
                        ui.output_plot("plot_barras"),
                    ),
                    col_widths=[7, 5]
                ),
                ui.card(
                    ui.card_header("Mapa de Calor Histórico (Emociones vs Años)"),
                    ui.output_plot("plot_heatmap"),
                ),
                style="padding: 10px;"
            )
        ),

        # TAB 2: Integración de Power BI
        ui.nav_panel(
            "Dashboard Power BI",
            ui.div(
                ui.h2("Reporte Ejecutivo Integrado", style="margin-top: 15px; margin-bottom: 20px;"),
                ui.card(
                    ui.tags.iframe(
                        # Integración segura de tu reporte específico de Power BI (autoAuth activa inicio de sesión automático)
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
    title="Sephora Sentiment Pro"
)

# ==========================================
# 3. LÓGICA DEL SERVIDOR (FAST REACTIVE)
# ==========================================

def server(input, output, session):

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
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#16161A')
        ax.set_facecolor('#16161A')
        
        df_f = filtered_time_data()
        
        for emo in df_f["emotion_es"].unique():
            temp = df_f[df_f["emotion_es"] == emo]
            color = COLOR_PALETTE.get(emo, '#FFFFFF')
            ax.plot(temp["date"], temp["count"], label=emo, color=color, linewidth=2.5)
            
        ax.set_title("Evolución de los Sentimientos por Año", color="#FFFFFF", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Línea de Tiempo", color="#A0A0B0", fontsize=10)
        ax.set_ylabel("Cantidad de Reseñas", color="#A0A0B0", fontsize=10)
        
        # Ajustar rejilla y ejes para el Dark Theme
        ax.grid(True, color="rgba(255, 255, 255, 0.05)", linestyle="--")
        ax.spines['bottom'].set_color('#2A2A35')
        ax.spines['left'].set_color('#2A2A35')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#A0A0B0')
        
        fig.autofmt_xdate(rotation=35)
        
        # Leyenda Premium
        leg = ax.legend(facecolor='#16161A', edgecolor='rgba(255, 255, 255, 0.1)')
        for text in leg.get_texts():
            text.set_color('#FFFFFF')
            
        plt.tight_layout()
        return fig

    # ---- GRÁFICA 2: Volumen de Emociones ----
    @output
    @render.plot
    def plot_barras():
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#16161A')
        ax.set_facecolor('#16161A')
        
        df_f = filtered_totals()
        
        if not df_f.empty:
            # Asignamos colores correspondientes
            colors = [COLOR_PALETTE.get(emo, '#FFFFFF') for emo in df_f["emotion_es"]]
            bars = ax.bar(df_f["emotion_es"], df_f["count"], color=colors, edgecolor='rgba(255,255,255,0.1)', width=0.6)
            
            # Agregar etiquetas de valor en las barras
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 puntos de desfase vertical
                            textcoords="offset points",
                            ha='center', va='bottom', color='#FFFFFF', fontsize=9)
                            
        ax.set_title("Volumen Crítico por Emoción", color="#FFFFFF", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Emoción Clasificada", color="#A0A0B0")
        ax.set_ylabel("Total Reseñas", color="#A0A0B0")
        
        ax.grid(True, axis='y', color="rgba(255, 255, 255, 0.05)", linestyle="--")
        ax.spines['bottom'].set_color('#2A2A35')
        ax.spines['left'].set_color('#2A2A35')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#A0A0B0')
        
        plt.tight_layout()
        return fig

    # ---- GRÁFICA 3: Mapa de Calor ----
    @output
    @render.plot
    def plot_heatmap():
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#16161A')
        ax.set_facecolor('#16161A')
        
        df_f = filtered_heatmap()
        
        if df_f.empty:
            ax.text(0.5, 0.5, "Selecciona al menos una emoción en el panel lateral.", 
                    horizontalalignment='center', verticalalignment='center', color='#FFFFFF', transform=ax.transAxes)
        else:
            # Pivotar datos para generar la matriz del Heatmap
            heat = df_f.pivot(index="emotion_es", columns="year", values="count").fillna(0)
            
            # Dibujar mapa de calor premium con colores oscuros
            sns.heatmap(heat, cmap="magma", ax=ax, annot=True, fmt=".0f", 
                        cbar_kws={'label': 'Conteo de Reseñas'}, annot_kws={"size": 10, "weight": "bold"})
            
            # Estilizar barra de colores
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='#FFFFFF')
            cbar.ax.tick_params(labelsize=10, labelcolor='#FFFFFF')
            cbar.set_label('Conteo de Reseñas', color='#FFFFFF', size=10)
            
            ax.set_title("Intensidad Emocional por Año", color="#FFFFFF", fontsize=12, fontweight='bold', pad=15)
            ax.set_ylabel("Emoción", color="#A0A0B0")
            ax.set_xlabel("Año de la Reseña", color="#A0A0B0")
            
            ax.tick_params(colors='#FFFFFF')
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        return fig

# ==========================================
# 4. INSTANCIACIÓN DE APLICACIÓN
# ==========================================
app = App(app_ui, server)

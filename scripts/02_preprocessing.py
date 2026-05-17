import pandas as pd
import numpy as np
import os
import ast
import csv
from sklearn.preprocessing import MultiLabelBinarizer

# Aumentar el límite de tamaño de campo para evitar errores con reseñas muy largas
csv.field_size_limit(1000000)

def preprocess_data():
    raw_path = "data/raw"
    processed_path = "data/processed"
    reviews_input = os.path.join(processed_path, "reviews_consolidated.csv")
    products_input = os.path.join(raw_path, "product_info.csv")
    output_path = os.path.join(processed_path, "cleaned_sephora_data.csv")
    
    print("🏗️ Iniciando Preprocesamiento de Grado Industrial (Paso 2)...")
    
    # 1. CARGA DE DATOS
    print("📥 Cargando datasets...")
    df_products = pd.read_csv(products_input, engine='python')
    df_reviews = pd.read_csv(reviews_input, engine='python', on_bad_lines='skip')

    # 2. FILTRADO EXTENDIDO (Basado en Notebook Original)
    palabras_relacionadas = [
        'Remover' , 'remover' , 'Cleanser' , 'cleanser' , 'Make up cleanser' , 'bb cream' , 'bbcream' , 'BB' ,
        'Mask' , 'Masks' , 'mask' , 'Lip balm' , 'lip balm' , 'balm' , 'Gel' , 'gel' , 'make up remover' ,
        'Make up remover' ,'Hairdresser', 'Curl','Repair','Thickening',"Hairdresser",'Heat' ,'shampoo','Shampoo',
        'hair','haircare','Haircare','Style','Styler','style','scalp','Scalp','Conditioner','conditioner','Frizz'
    ]
    
    print(f"🔍 Filtrando productos relevantes ({len(palabras_relacionadas)} términos)...")
    df_products = df_products[df_products['product_name'].str.contains('|'.join(palabras_relacionadas), case=False, na=False)]
    print(f"📦 Productos seleccionados: {len(df_products)}")

    # 3. LIMPIEZA DE PRODUCTOS (Lógica de Notebook)
    print("🧹 Limpiando metadatos de productos...")
    
    # Convertir 'highlights' de string a lista real
    def safe_literal_eval(val):
        try:
            return ast.literal_eval(val) if pd.notnull(val) else []
        except:
            return []

    df_products['highlights'] = df_products['highlights'].apply(safe_literal_eval)

    # Binarizar Highlights (MultiLabelBinarizer)
    mlb = MultiLabelBinarizer()
    highlights_encoded = pd.DataFrame(
        mlb.fit_transform(df_products['highlights']),
        columns=mlb.classes_,
        index=df_products.index
    )
    
    # Unir highlights binarizados y ELIMINAR la columna original de listas
    # Esto evita el error "unhashable type: 'list'" al cruzar o eliminar duplicados
    df_products = pd.concat([df_products, highlights_encoded], axis=1).drop(columns=['highlights'])
    
    # Conversión a booleanos para Power BI
    bool_cols = ['limited_edition', 'new', 'online_only', 'out_of_stock', 'sephora_exclusive']
    for col in bool_cols:
        df_products[col] = df_products[col].astype(bool)

    # Imputación de nulos en rating y loves (usando mediana según notebook)
    df_products['rating'] = df_products['rating'].fillna(df_products['rating'].median())
    df_products['loves_count'] = df_products['loves_count'].fillna(0)

    # 4. CRUCE CON REVIEWS
    print("🤝 Cruzando reseñas con productos filtrados...")
    # Solo reviews de los productos que pasaron el filtro anterior
    # Usamos product_id como llave primaria para el cruce
    df_final = pd.merge(df_reviews, df_products, on='product_id', how='inner', suffixes=('', '_prod'))

    # 5. LIMPIEZA FINAL DE REVIEWS
    print("✨ Refinando dataset final...")
    df_final['submission_time'] = pd.to_datetime(df_final['submission_time'], errors='coerce')
    df_final = df_final.dropna(subset=['review_text'])
    
    # Eliminar duplicados técnicos si existen
    df_final = df_final.drop_duplicates()

    print(f"✅ Preprocesamiento finalizado con éxito.")
    print(f"📊 Registros totales: {len(df_final):,}")
    print(f"🏷️ Columnas generadas (incluyendo tags): {len(df_final.columns)}")

    # 6. GUARDAR
    df_final.to_csv(output_path, index=False)
    print(f"💾 Archivo Maestro guardado: {output_path}")

if __name__ == "__main__":
    preprocess_data()

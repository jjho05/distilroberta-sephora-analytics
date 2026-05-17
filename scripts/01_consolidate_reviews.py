import pandas as pd
import glob
import os

def consolidate_reviews():
    # Rutas relativas para portabilidad
    raw_path = "data/raw"
    output_path = "data/processed/reviews_consolidated.csv"
    
    print("🚀 Iniciando Consolidación de Datasets...")
    
    # Buscamos todos los archivos de reviews (excluyendo product_info)
    review_files = glob.glob(os.path.join(raw_path, "reviews_*.csv"))
    print(f"📂 Archivos encontrados: {len(review_files)}")
    
    df_list = []
    
    for file in sorted(review_files):
        print(f"📄 Leyendo: {os.path.basename(file)}...")
        # Usamos low_memory=False para evitar warnings de tipos mixtos en datasets grandes
        temp_df = pd.read_csv(file, low_memory=False)
        df_list.append(temp_df)
    
    if not df_list:
        print("❌ No se encontraron archivos de reviews.")
        return

    # Concatenación
    print("🔗 Uniendo datasets...")
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Limpieza básica de índices duplicados que suelen venir en Kaggle
    if 'Unnamed: 0' in full_df.columns:
        full_df = full_df.drop(columns=['Unnamed: 0'])
    
    total_rows = len(full_df)
    print(f"✅ Consolidación exitosa. Total de registros: {total_rows:,}")
    
    # Guardar resultado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"💾 Archivo guardado en: {output_path}")

if __name__ == "__main__":
    consolidate_reviews()

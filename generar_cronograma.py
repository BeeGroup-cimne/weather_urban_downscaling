import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

# --- DATOS DEL CRONOGRAMA ---
tasks = [
    # AÑO 1: INGENIERÍA
    dict(Task="Revisión Bibliográfica (SOTA)", Start='2025-10-01', Finish='2026-03-30', Phase='Año 1: Fundamentos'),
    dict(Task="Ingeniería de Datos (Zarr/ERA5)", Start='2026-03-01', Finish='2026-09-30', Phase='Año 1: Fundamentos'),
    dict(Task="Dockerización (Dual Engine)", Start='2026-04-01', Finish='2026-07-30', Phase='Año 1: Fundamentos'),
    dict(Task="Baseline U-Net (TensorFlow)", Start='2026-08-01', Finish='2026-12-31', Phase='Año 1: Fundamentos'),
    
    # AÑO 2: INVESTIGACIÓN NUCLEAR
    dict(Task="Implementación Mamba (PyTorch)", Start='2027-01-01', Finish='2027-05-30', Phase='Año 2: Investigación'),
    dict(Task="Ablation Studies (HPC/Cloud)", Start='2027-04-01', Finish='2027-09-30', Phase='Año 2: Investigación'),
    dict(Task="Paper 1: Metodología & Arquitectura", Start='2027-06-01', Finish='2027-10-30', Phase='Año 2: Investigación'),
    dict(Task="Estancia Internacional (3 Meses)", Start='2027-09-01', Finish='2027-12-31', Phase='Año 2: Investigación'),
    
    # AÑO 3: VALIDACIÓN Y TESIS
    dict(Task="Validación Física (Extremos)", Start='2028-01-01', Finish='2028-03-30', Phase='Año 3: Cierre'),
    dict(Task="Paper 2: Aplicación Climática", Start='2028-04-01', Finish='2028-05-30', Phase='Año 3: Cierre'),
    dict(Task="Escritura Memoria Tesis", Start='2028-06-01', Finish='2028-07-15', Phase='Año 3: Cierre'),
    dict(Task="Depósito y Defensa", Start='2028-07-15', Finish='2028-08-31', Phase='Año 3: Cierre'),
]

# Convertir a DataFrame
df = pd.DataFrame(tasks)
df['Start'] = pd.to_datetime(df['Start'])
df['Finish'] = pd.to_datetime(df['Finish'])
df['Duration'] = df['Finish'] - df['Start']

# --- CONFIGURACIÓN DEL GRÁFICO ---
fig, ax = plt.subplots(figsize=(12, 8))

# Colores por fase
colors = {'Año 1: Fundamentos': '#1f77b4', 'Año 2: Investigación': '#ff7f0e', 'Año 3: Cierre': '#2ca02c'}

# Dibujar barras
for i, task in df.iterrows():
    start = mdates.date2num(task['Start'])
    end = mdates.date2num(task['Finish'])
    duration = end - start
    
    # Barra
    ax.barh(task['Task'], duration, left=start, color=colors[task['Phase']], edgecolor='black', alpha=0.8)
    
    # Texto de fechas (Opcional, para detalle)
    # ax.text(start + duration/2, i, f"{task['Duration'].days}d", ha='center', va='center', color='white', fontsize=8)

# Formato de Ejes
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

# Etiquetas y Título
ax.set_xlabel('Calendario')
ax.set_title('Cronograma de Tesis Doctoral: Weather Downscaling & AI (2025-2028)', fontsize=14, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Invertir eje Y para que el Año 1 salga arriba
plt.gca().invert_yaxis()

# --- GUARDAR ---
output_file = 'cronograma_tesis_final.png'
plt.savefig(output_file, dpi=300)
print(f"✅ Imagen generada: {output_file}")
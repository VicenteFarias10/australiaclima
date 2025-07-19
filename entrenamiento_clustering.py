import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("Cargando datos...")
df = pd.read_csv('wheaterPba3Completo.csv')

print("Preparando datos para clustering...")
# Seleccionar variables para clustering
features = ['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm', 'Rainfall', 'Evaporation']
df_clustering = df[features].dropna()

print(f"Datos disponibles para clustering: {len(df_clustering)} registros")

# Escalar los datos
print("Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering)

# Entrenar modelo K-Means
print("Entrenando modelo K-Means...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Guardar modelo
print("Guardando modelo de clustering...")
joblib.dump(kmeans, 'modelo_entrenado.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Modelo de clustering guardado.")

# Crear visualización de ejemplo
# print("Creando visualización de ejemplo...")
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Para evitar cargar todo el df en memoria para el plot, tomamos una muestra
# df_sample = df_clustering.sample(n=min(1000, len(df_clustering)), random_state=42)
# df_sample['cluster'] = kmeans.predict(scaler.transform(df_sample[features]))
#
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_sample, x='MaxTemp', y='Humidity3pm', hue='cluster', palette='viridis', s=100, alpha=0.8)
# plt.title('Visualización de Clusters Climáticos (Muestra)')
# plt.xlabel('Temperatura Máxima (°C)')
# plt.ylabel('Humedad a las 3pm (%)')
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.savefig('analisis_clustering.png', dpi=300, bbox_inches='tight')
# plt.close()

print("\n--- Entrenamiento de clustering completado ---")
print("Archivos generados:")
print("- ../models/clustering/modelo_entrenado.pkl: Modelo entrenado")
print("- ../models/clustering/scaler.pkl: Scaler")
print("- ../models/clustering/analisis_clustering.png: Visualización de ejemplo") 
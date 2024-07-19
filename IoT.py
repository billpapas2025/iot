import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Importación de Bibliotecas
st.title('Integración de Big Data y Machine Learning en IoT')
st.write("""
Esta aplicación muestra cómo se pueden integrar técnicas de Big Data y Machine Learning en aplicaciones de IoT.
""")

# 2. Cargar y Preprocesar Datos
@st.cache_data
def load_data():
    # Simulamos un conjunto de datos de IoT
    data = pd.DataFrame({
        'sensor_1': np.random.rand(1000),
        'sensor_2': np.random.rand(1000),
        'sensor_3': np.random.rand(1000),
        'output': np.random.rand(1000) * 100
    })
    return data

data = load_data()
st.write("### Datos de Sensores de IoT", data.head())

# Mostrar correlación entre variables
st.write("### Mapa de Calor de Correlación")
st.write(data.corr())

# 3. Entrenamiento del Modelo de Machine Learning
X = data[['sensor_1', 'sensor_2', 'sensor_3']]
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Evaluación del Modelo")
st.write(f"**Error Cuadrático Medio (MSE):** {mse}")
st.write(f"**Coeficiente de Determinación (R2):** {r2}")

# 4. Visualización y Predicción en Tiempo Real
st.write("### Comparación de Valores Reales vs Predicciones")
comparison = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
st.line_chart(comparison)

# Interfaz para predicciones en tiempo real
st.write("### Predicciones en Tiempo Real")
sensor_1 = st.slider('Valor del Sensor 1', 0.0, 1.0, 0.5)
sensor_2 = st.slider('Valor del Sensor 2', 0.0, 1.0, 0.5)
sensor_3 = st.slider('Valor del Sensor 3', 0.0, 1.0, 0.5)

input_data = np.array([[sensor_1, sensor_2, sensor_3]])
prediction = model.predict(input_data)

st.write(f"**Predicción del Modelo:** {prediction[0]}")

# 5. Ejecución de la Aplicación
# Para ejecutar esta aplicación, guarda el código en un archivo llamado `app.py` y ejecuta el siguiente comando en la terminal:
# streamlit run app.py

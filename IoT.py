import streamlit as st
import pandas as pd
import numpy as np

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
X = data[['sensor_1', 'sensor_2', 'sensor_3']].values
y = data['output'].values

# Agregamos una columna de unos a X para el término independiente
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Dividimos los datos en entrenamiento y prueba manualmente
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
train_size = int(0.8 * X.shape[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Calculamos los coeficientes de la regresión lineal
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predicciones
y_pred = X_test @ coefficients

# Evaluación del modelo
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

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

input_data = np.array([1, sensor_1, sensor_2, sensor_3])
prediction = input_data @ coefficients

st.write(f"**Predicción del Modelo:** {prediction}")

# 5. Ejecución de la Aplicación
# Para ejecutar esta aplicación, guarda el código en un archivo llamado `app.py` y ejecuta el siguiente comando en la terminal:
# streamlit run app.py

# 🛡️ FraudML: Inteligencia en Detección Financiera

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)
![CRISP-ML](https://img.shields.io/badge/Methodology-CRISP--ML-green.svg)

Este proyecto implementa el ciclo de vida completo de un modelo de Machine Learning para la detección de fraude en quejas financieras en Colombia.

## 🚀 Pipeline de Datos (3 Cuadernos)

El proceso se divide en tres cuadernos detallados para mayor claridad y modularidad:

1.  **[01_ETL_Fraude.ipynb](01_ETL_Fraude.ipynb)**: Extracción de datos, limpieza profunda, manejo de nulos y generación de la variable objetivo. Genera el archivo `fraude_limpio.csv`.
2.  **[02_EDA_Fraude.ipynb](02_EDA_Fraude.ipynb)**: Análisis exploratorio visual, identificación de patrones geográficos y por producto.
3.  **[03_Model_Fraude.ipynb](03_Model_Fraude.ipynb)**: Codificación de variables, entrenamiento de Regresión Logística y evaluación de métricas.

## 📦 Dataset
- `fraude.csv`: Dataset original.
- `fraude_limpio.csv`: Dataset procesado y listo para modelado.

## 🛠️ Despliegue e Interfaz Modular
La aplicación central (`app.py`) ha sido potenciada con un panel de navegación lateral (sidebar) que incluye:

- **🏠 Inicio**: Dashboard de bienvenida con métricas clave y vista del dataset.
- **📊 Análisis Estadístico**: Profundización matemática de las variables.
- **📈 Gráficos Interactivos**: Visualizaciones avanzadas (Caja de bigotes, Histogramas, Barras).
- **🤖 Modelo Predictivo**: Formulario inteligente con efectos de celebración (bombas y platillos) tras la predicción.
- **[index.html](index.html)**: Landing page premium con la explicación teórica y diccionario de datos.

## 💻 Requerimientos
- **Versión de Python recomendada**: 3.9 o superior.
- **Instalación**:
  ```bash
  pip install -r requirements.txt
  ```
- **Ejecución de la App**:
  ```bash
  streamlit run app.py
  ```

---
Basado en la metodología **CRISP-ML** para garantizar un proceso de desarrollo estructurado y profesional.

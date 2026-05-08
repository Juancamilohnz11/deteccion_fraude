import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Fraud Detector Pro", layout="wide", initial_sidebar_state="expanded")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .stSidebar {
        background-color: #1e293b;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(45deg, #6366f1, #ec4899);
        color: white;
        font-weight: bold;
        border: none;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to load data and assets
@st.cache_resource
def load_assets():
    model, encoders, df = None, None, None
    if os.path.exists('fraud_model.pkl') and os.path.exists('encoders.pkl'):
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
    if os.path.exists('fraude_limpio.csv'):
        df = pd.read_csv('fraude_limpio.csv')
    return model, encoders, df

model, encoders, df = load_assets()

# Sidebar Navigation
st.sidebar.title("💎 Navegación")
menu = st.sidebar.radio(
    "Seleccione una sección:",
    ["🏠 Inicio", "📊 Análisis Estadístico", "📈 Gráficos Interactivos", "🤖 Modelo Predictivo"]
)

# --- SECCIÓN: INICIO ---
if menu == "🏠 Inicio":
    st.title("🛡️ Ecosistema de Detección de Fraude")
    st.write("Bienvenido al panel de control inteligente. Aquí podrá monitorear y predecir anomalías financieras utilizando IA.")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Quejas", len(df))
        with col2:
            st.metric("Fraudes Detectados", len(df[df['es_fraude'] == 1]))
        with col3:
            st.metric("Tasa de Fraude", f"{(len(df[df['es_fraude'] == 1])/len(df)*100):.2f}%")
        with col4:
            st.metric("Entidades Monitoreadas", df['tipo_entidad'].nunique())
            
        st.markdown("### Vista Previa del Dataset Limpio")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("No se encontró el archivo 'fraude_limpio.csv'. Por favor ejecute el ETL primero.")

# --- SECCIÓN: ANÁLISIS ESTADÍSTICO ---
elif menu == "📊 Análisis Estadístico":
    st.title("📊 Profundización Estadística")
    if df is not None:
        st.subheader("Descripción Matemática")
        st.write(df.describe())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Conteo por Tipo de Entidad")
            st.write(df['tipo_entidad'].value_counts())
        with col2:
            st.subheader("Conteo por Instancia")
            st.write(df['instancia_recepcion'].value_counts())
    else:
        st.error("Datos no disponibles.")

# --- SECCIÓN: GRÁFICOS ---
elif menu == "📈 Gráficos Interactivos":
    st.title("📈 Visualización de Datos (EDA)")
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📊 Barras e Histogramas", "📦 Caja y Bigotes", "🔥 Correlaciones"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Histograma de Quejas")
                fig, ax = plt.subplots()
                sns.histplot(df['cantidad_quejas_recibidas'], bins=20, kde=True, ax=ax, color='#6366f1')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Fraude por Producto (Top 5)")
                fig, ax = plt.subplots()
                top_products = df[df['es_fraude'] == 1]['producto'].value_counts().head(5)
                top_products.plot(kind='bar', ax=ax, color='#ec4899')
                st.pyplot(fig)
        
        with tab2:
            st.subheader("Caja de Bigotes: Distribución de Quejas")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df, x='es_fraude', y='cantidad_quejas_recibidas', ax=ax, palette='Set2')
            ax.set_xticklabels(['No Fraude', 'Fraude'])
            st.pyplot(fig)
            st.info("Este gráfico permite identificar valores atípicos (outliers) en el volumen de quejas fraudulentas.")

        with tab3:
            st.subheader("Matriz de Frecuencia de Fraude por Departamento")
            pivot = df.groupby('departamento')['es_fraude'].sum().sort_values(ascending=False).head(10)
            st.bar_chart(pivot)
    else:
        st.error("Datos no disponibles.")

# --- SECCIÓN: MODELO ---
elif menu == "🤖 Modelo Predictivo":
    st.title("🤖 Modelado y Predicción")
    if model is None or encoders is None:
        st.error("Modelo o Encoders no encontrados. Verifique los archivos .pkl")
    else:
        with st.form("prediction_form"):
            st.subheader("Ingresar Datos para Predicción")
            col1, col2 = st.columns(2)
            
            with col1:
                tipo = st.selectbox("Tipo de Entidad", encoders['tipo_entidad'].classes_)
                instancia = st.selectbox("Instancia de Recepción", encoders['instancia_recepcion'].classes_)
                cod_entidad = st.number_input("Código de Entidad", value=7)
            
            with col2:
                prod = st.selectbox("Producto Financiero", encoders['producto'].classes_)
                depto = st.selectbox("Departamento", encoders['departamento'].classes_)
                cant = st.number_input("Cantidad de quejas", min_value=1, value=1)
            
            predict_btn = st.form_submit_button("EJECUTAR MODELO 🚀")
            
            if predict_btn:
                # Prepare data
                input_df = pd.DataFrame({
                    'tipo_entidad': [tipo],
                    'codigo_entidad': [cod_entidad],
                    'instancia_recepcion': [instancia],
                    'producto': [prod],
                    'departamento': [depto],
                    'cantidad_quejas_recibidas': [cant]
                })
                
                # Encode
                for col in ['tipo_entidad', 'instancia_recepcion', 'producto', 'departamento']:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                
                # Predict
                prob = model.predict_proba(input_df)[0][1]
                pred = model.predict(input_df)[0]
                
                st.markdown("### Resultado del Análisis")
                if pred == 1:
                    st.error(f"🚨 ALERTA DE FRAUDE DETECTADA - Probabilidad: {prob:.2%}")
                    st.snow()
                else:
                    st.success(f"✅ OPERACIÓN NORMAL - Probabilidad de Fraude: {prob:.2%}")
                    st.balloons()
                
                st.write("El modelo de Regresión Logística ha procesado los datos satisfactoriamente.")

st.sidebar.markdown("---")
st.sidebar.caption("Proyecto FraudML - Metodología CRISP-ML")

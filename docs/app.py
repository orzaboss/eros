import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import io
import tempfile
import os

# Configuración de la página
st.set_page_config(
    page_title="🌌 Predictor de Trayectorias de Asteroides - TensorFlow",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 10px 0;
    }
    .stMetric {
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🌌 Predictor de Trayectorias de Asteroides</h1>
    <p>Análisis y predicción de órbitas usando TensorFlow/Keras</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración del Modelo")
LOOKBACK = st.sidebar.slider("📊 Pasos históricos (LOOKBACK)", 5, 30, 10, 
                           help="Cantidad de puntos históricos para predecir")
PREDICTION = st.sidebar.slider("🎯 Pasos a predecir (PREDICTION)", 1, 20, 10,
                              help="Cantidad de pasos futuros a predecir")

st.sidebar.markdown("---")
st.sidebar.header("🧠 Configuración de Red Neuronal")
hidden_units_1 = st.sidebar.slider("Neuronas capa 1", 32, 256, 128, step=32)
hidden_units_2 = st.sidebar.slider("Neuronas capa 2", 16, 128, 64, step=16)
epochs = st.sidebar.slider("Épocas de entrenamiento", 10, 100, 20, step=10)
batch_size = st.sidebar.selectbox("Tamaño de batch", [16, 32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("Tasa de aprendizaje", [0.001, 0.01, 0.1], index=0)

# Funciones auxiliares (manteniendo la lógica original)
@st.cache_data
def extract_datetime(text):
    """Extrae fecha y hora usando regex"""
    match = re.search(r'\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+', str(text))
    if match:
        return match.group(0)
    else:
        return None

@st.cache_data
def load_and_clean_from_upload(file_content, filename):
    """Carga y limpia archivo subido (versión original)"""
    try:
        # Decodificar contenido
        content = file_content.decode('utf-8')
        lines = content.split('\n')
        
        # Crear DataFrame temporal
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write('\n'.join(lines[79:]))  # Saltear header
            tmp_path = tmp_file.name
        
        # Leer con pandas (lógica original)
        df_raw = pd.read_csv(tmp_path, sep=',\\s*', engine='python', 
                            header=None, on_bad_lines='skip')
        
        # Limpiar archivo temporal
        os.unlink(tmp_path)
        
        if df_raw.shape[1] < 10:
            st.warning(f"Archivo {filename} tiene menos columnas de las esperadas.")
            return pd.DataFrame()
        
        # Extraer datetime (lógica original)
        df_raw['datetime_str'] = df_raw[1].apply(extract_datetime)
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime_str'], errors='coerce')
        
        # Extraer características numéricas
        df_numeric = df_raw.iloc[:, 4:10].copy()
        df_numeric.columns = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
        df_numeric['datetime'] = df_raw['datetime']
        
        # Convertir a numérico
        for col in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        
        # Eliminar NaN
        df_numeric = df_numeric.dropna().reset_index(drop=True)
        return df_numeric
        
    except Exception as e:
        st.error(f"Error procesando {filename}: {e}")
        return pd.DataFrame()

@st.cache_data
def scale_features(df):
    """Escala características (lógica original)"""
    scalers = {}
    df_scaled = pd.DataFrame()
    
    for col in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
        scaler = MinMaxScaler()
        df_scaled[col] = scaler.fit_transform(df[[col]]).flatten()
        scalers[col] = scaler
    
    df_scaled['datetime'] = df['datetime']
    return df_scaled, scalers

def create_sequences(df, lookback, prediction):
    """Crear secuencias (lógica original TensorFlow)"""
    data = df[['X','Y','Z','VX','VY','VZ']].values
    X, y = [], []
    for i in range(len(data) - lookback - prediction + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+prediction])
    return np.array(X), np.array(y)

def create_tensorflow_model(lookback, prediction, hidden_1, hidden_2, lr):
    """Crear modelo TensorFlow (arquitectura original)"""
    model = Sequential([
        Flatten(input_shape=(lookback, 6)),
        Dense(hidden_1, activation='relu'),
        Dense(hidden_2, activation='relu'),
        Dense(prediction * 6)  # salida para todos los pasos y features
    ])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

def predict_and_descale(model, X_input, scalers, prediction_steps):
    """Predecir y desescalar (lógica original)"""
    y_pred_flat = model.predict(X_input, verbose=0)
    y_pred = y_pred_flat.reshape((-1, prediction_steps, 6))
    y_pred_descaled = np.empty_like(y_pred)
    
    for i, feat in enumerate(['X','Y','Z','VX','VY','VZ']):
        scaler = scalers[feat]
        for step in range(prediction_steps):
            y_pred_descaled[:, step, i] = scaler.inverse_transform(
                y_pred[:, step, i].reshape(-1,1)
            ).flatten()
    
    return y_pred_descaled

def create_3d_trajectory_plot(y_true, y_pred, asteroid_name, step):
    """Crear gráfico 3D de trayectorias"""
    fig = go.Figure()
    
    # Limitar puntos para mejor rendimiento
    max_points = min(200, len(y_true))
    idx = np.linspace(0, len(y_true)-1, max_points, dtype=int)
    
    # Trayectoria real
    fig.add_trace(go.Scatter3d(
        x=y_true[idx, step, 0],
        y=y_true[idx, step, 1],
        z=y_true[idx, step, 2],
        mode='markers+lines',
        name='🎯 Trayectoria Real',
        marker=dict(size=4, color='blue', symbol='circle'),
        line=dict(width=3, color='blue')
    ))
    
    # Trayectoria predicha
    fig.add_trace(go.Scatter3d(
        x=y_pred[idx, step, 0],
        y=y_pred[idx, step, 1],
        z=y_pred[idx, step, 2],
        mode='markers+lines',
        name='🤖 Trayectoria Predicha',
        marker=dict(size=4, color='red', symbol='diamond'),
        line=dict(width=3, color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'🌌 {asteroid_name} - Posiciones 3D (Paso +{step+1})',
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            bgcolor='rgba(0,0,0,0.05)'
        ),
        height=600,
        showlegend=True
    )
    
    return fig

def create_velocity_comparison_plot(y_true, y_pred, asteroid_name, step):
    """Crear gráfico de comparación de velocidades"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Velocidad X', 'Velocidad Y', 'Velocidad Z', 'Error por Componente'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.1
    )
    
    # Limitar puntos
    max_points = min(100, len(y_true))
    idx = np.linspace(0, len(y_true)-1, max_points, dtype=int)
    
    velocity_features = ['VX', 'VY', 'VZ']
    colors = ['red', 'green', 'blue']
    
    for i, (feat, color) in enumerate(zip(velocity_features, colors)):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        if i < 3:  # VX, VY, VZ
            # Real
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(idx))),
                    y=y_true[idx, step, i+3],
                    name=f'Real {feat}',
                    line=dict(color=color, width=2),
                    mode='lines+markers'
                ),
                row=row, col=col
            )
            
            # Predicción
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(idx))),
                    y=y_pred[idx, step, i+3],
                    name=f'Pred {feat}',
                    line=dict(color=color, width=2, dash='dash'),
                    mode='lines+markers'
                ),
                row=row, col=col
            )
    
    # Gráfico de barras de error
    errors = []
    for i in range(3):
        error = np.mean(np.abs(y_true[:, step, i+3] - y_pred[:, step, i+3]))
        errors.append(error)
    
    fig.add_trace(
        go.Bar(
            x=velocity_features,
            y=errors,
            name='Error Absoluto Medio',
            marker_color=['red', 'green', 'blue']
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'🚀 {asteroid_name} - Análisis de Velocidades (Paso +{step+1})',
        height=700,
        showlegend=True
    )
    
    return fig

def create_error_evolution_plot(y_true, y_pred, asteroid_name, prediction_steps):
    """Crear gráfico de evolución del error"""
    steps = list(range(1, prediction_steps + 1))
    rmse_values = []
    mae_values = []
    
    for step in range(prediction_steps):
        rmse = np.sqrt(mean_squared_error(y_true[:, step, :], y_pred[:, step, :]))
        mae = mean_absolute_error(y_true[:, step, :], y_pred[:, step, :])
        rmse_values.append(rmse)
        mae_values.append(mae)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps, y=rmse_values,
        name='RMSE',
        line=dict(color='red', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=steps, y=mae_values,
        name='MAE',
        line=dict(color='blue', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'📊 {asteroid_name} - Evolución del Error por Paso',
        xaxis_title='Paso de Predicción',
        yaxis_title='Error',
        height=400,
        showlegend=True
    )
    
    return fig

# Interfaz principal
st.header("📁 Carga de Datos")

uploaded_files = st.file_uploader(
    "Sube archivos de asteroides (.txt)",
    type=['txt'],
    accept_multiple_files=True,
    help="Archivos con datos orbitales de asteroides en formato estándar"
)

if uploaded_files:
    asteroid_data = {}
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Procesando {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        file_content = uploaded_file.read()
        df = load_and_clean_from_upload(file_content, uploaded_file.name)
        
        if not df.empty:
            asteroid_name = uploaded_file.name.replace('.txt', '')
            asteroid_data[asteroid_name] = df
            st.success(f"✅ {asteroid_name}: {len(df)} puntos de datos cargados")
        else:
            st.error(f"❌ Error procesando {uploaded_file.name}")
    
    progress_bar.empty()
    status_text.empty()
    
    if asteroid_data:
        st.markdown("---")
        st.header("🧠 Entrenamiento del Modelo TensorFlow")
        
        # Mostrar información del dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Asteroides cargados", len(asteroid_data))
        with col2:
            total_points = sum(len(df) for df in asteroid_data.values())
            st.metric("📈 Total puntos de datos", total_points)
        with col3:
            st.metric("🔄 Lookback configurado", LOOKBACK)
        with col4:
            st.metric("🎯 Predicción configurada", PREDICTION)
        
        # Preparar datos para entrenamiento
        if st.button("🚀 Entrenar Modelo TensorFlow", type="primary"):
            with st.spinner("Preparando datos para entrenamiento..."):
                X_train_list = []
                y_train_list = []
                scalers_global = {}
                
                for name, df in asteroid_data.items():
                    df_scaled, scalers = scale_features(df)
                    scalers_global[name] = scalers
                    
                    X_seq, y_seq = create_sequences(df_scaled, LOOKBACK, PREDICTION)
                    
                    if X_seq.shape[0] > 0:
                        X_train_list.append(X_seq)
                        y_train_list.append(y_seq)
                        st.write(f"**{name}**: {X_seq.shape[0]} secuencias creadas")
                
                if X_train_list:
                    X_train = np.concatenate(X_train_list, axis=0)
                    y_train = np.concatenate(y_train_list, axis=0)
                    
                    st.success(f"✅ Dataset preparado: {X_train.shape[0]} secuencias totales")
                    
                    # Crear y entrenar modelo
                    with st.spinner("Entrenando red neuronal..."):
                        model = create_tensorflow_model(
                            LOOKBACK, PREDICTION, 
                            hidden_units_1, hidden_units_2, 
                            learning_rate
                        )
                        
                        # Preparar datos para TensorFlow
                        y_train_flat = y_train.reshape((y_train.shape[0], PREDICTION * 6))
                        
                        # Entrenar con callback para mostrar progreso
                        history = model.fit(
                            X_train, y_train_flat,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            verbose=0
                        )
                        
                        st.success("🎉 Modelo entrenado exitosamente!")
                        
                        # Mostrar métricas de entrenamiento
                        col1, col2 = st.columns(2)
                        with col1:
                            final_loss = history.history['loss'][-1]
                            st.metric("🎯 Loss final", f"{final_loss:.6f}")
                        with col2:
                            final_mae = history.history['mae'][-1]
                            st.metric("📊 MAE final", f"{final_mae:.6f}")
                        
                        # Gráfico de pérdida
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=history.history['loss'],
                            name='Training Loss',
                            line=dict(color='blue')
                        ))
                        fig_loss.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            name='Validation Loss',
                            line=dict(color='red')
                        ))
                        fig_loss.update_layout(
                            title='📈 Evolución de la Pérdida durante Entrenamiento',
                            xaxis_title='Época',
                            yaxis_title='Loss (MSE)'
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
                        # Guardar en session state
                        st.session_state['model'] = model
                        st.session_state['scalers'] = scalers_global
                        st.session_state['asteroid_data'] = asteroid_data
                        st.session_state['X_train'] = X_train
                        st.session_state['y_train'] = y_train
                        st.session_state['history'] = history
        
        # Evaluación y visualización
        if 'model' in st.session_state:
            st.markdown("---")
            st.header("📊 Evaluación y Visualización")
            
            selected_asteroid = st.selectbox(
                "Selecciona un asteroide para evaluar:",
                list(asteroid_data.keys()),
                help="Elige el asteroide que deseas analizar en detalle"
            )
            
            if selected_asteroid:
                df_selected = asteroid_data[selected_asteroid]
                df_scaled, scalers = scale_features(df_selected)
                X_test, y_test = create_sequences(df_scaled, LOOKBACK, PREDICTION)
                
                if X_test.shape[0] > 0:
                    model = st.session_state['model']
                    
                    # Hacer predicciones
                    with st.spinner("Realizando predicciones..."):
                        y_pred_descaled = predict_and_descale(
                            model, X_test, scalers, PREDICTION
                        )
                        
                        # Desescalar datos reales
                        y_true_descaled = np.empty_like(y_test)
                        for i, feat in enumerate(['X','Y','Z','VX','VY','VZ']):
                            scaler = scalers[feat]
                            for step in range(PREDICTION):
                                y_true_descaled[:, step, i] = scaler.inverse_transform(
                                    y_test[:, step, i].reshape(-1,1)
                                ).flatten()
                    
                    # Métricas de evaluación
                    st.subheader("📈 Métricas de Rendimiento")
                    
                    # Crear métricas por paso
                    metrics_data = []
                    for step in range(PREDICTION):
                        rmse = np.sqrt(mean_squared_error(
                            y_true_descaled[:, step, :], 
                            y_pred_descaled[:, step, :]
                        ))
                        mae = mean_absolute_error(
                            y_true_descaled[:, step, :], 
                            y_pred_descaled[:, step, :]
                        )
                        metrics_data.append({
                            'Paso': step + 1,
                            'RMSE': rmse,
                            'MAE': mae
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(metrics_df, use_container_width=True)
                    with col2:
                        # Gráfico de evolución del error
                        fig_error = create_error_evolution_plot(
                            y_true_descaled, y_pred_descaled, 
                            selected_asteroid, PREDICTION
                        )
                        st.plotly_chart(fig_error, use_container_width=True)
                    
                    # Controles de visualización
                    st.subheader("🎮 Controles de Visualización")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        step_to_plot = st.selectbox(
                            "Paso a visualizar:",
                            list(range(1, PREDICTION + 1)),
                            index=0
                        ) - 1
                    
                    with col2:
                        show_3d = st.checkbox("Mostrar gráfico 3D", value=True)
                        show_velocity = st.checkbox("Mostrar análisis de velocidades", value=True)
                    
                    # Visualizaciones
                    if show_3d:
                        st.subheader("🌌 Visualización 3D de Trayectorias")
                        fig_3d = create_3d_trajectory_plot(
                            y_true_descaled, y_pred_descaled, 
                            selected_asteroid, step_to_plot
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    
                    if show_velocity:
                        st.subheader("🚀 Análisis de Velocidades")
                        fig_vel = create_velocity_comparison_plot(
                            y_true_descaled, y_pred_descaled, 
                            selected_asteroid, step_to_plot
                        )
                        st.plotly_chart(fig_vel, use_container_width=True)
                    
                    # Información adicional
                    st.subheader("ℹ️ Información del Modelo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🧠 Arquitectura", f"{hidden_units_1}-{hidden_units_2}")
                    with col2:
                        st.metric("📊 Épocas", epochs)
                    with col3:
                        st.metric("🎯 Batch Size", batch_size)
                    with col4:
                        st.metric("📈 Learning Rate", learning_rate)
                    
                    # Estadísticas del dataset
                    with st.expander("📊 Estadísticas del Dataset"):
                        st.subheader(f"Estadísticas para {selected_asteroid}")
                        st.dataframe(df_selected[['X', 'Y', 'Z', 'VX', 'VY', 'VZ']].describe())
                    
                    # Exportar resultados
                    st.subheader("💾 Exportar Resultados")
                    
                    if st.button("Descargar Métricas CSV"):
                        csv = metrics_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar CSV",
                            data=csv,
                            file_name=f"{selected_asteroid}_metrics.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("❌ No hay suficientes datos para crear secuencias de prueba")

else:
    # Información cuando no hay archivos cargados
    st.info("📁 Sube archivos de asteroides para comenzar el análisis")
    
    # Mostrar ejemplo de datos
    st.subheader("📋 Formato de Datos Esperado")
    
    example_data = {
        'Característica': ['X', 'Y', 'Z', 'VX', 'VY', 'VZ'],
        'Descripción': [
            'Posición X (AU)',
            'Posición Y (AU)', 
            'Posición Z (AU)',
            'Velocidad X (AU/día)',
            'Velocidad Y (AU/día)',
            'Velocidad Z (AU/día)'
        ],
        'Ejemplo': [1.234, 2.345, 3.456, 0.123, 0.234, 0.345]
    }
    
    st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    # Ayuda e información
    with st.expander("📖 Guía de Uso"):
        st.markdown("""
        ### 🚀 Cómo usar esta aplicación:
        
        1. **📁 Carga archivos**: Sube archivos .txt con datos orbitales
        2. **⚙️ Configura parámetros**: Ajusta LOOKBACK, PREDICTION y parámetros de la red neuronal
        3. **🧠 Entrena modelo**: Crea una red neuronal profunda con TensorFlow
        4. **📊 Evalúa resultados**: Visualiza predicciones y métricas
        5. **💾 Exporta datos**: Descarga resultados en CSV
        
        ### 📋 Características del modelo:
        - **Framework**: TensorFlow/Keras
        - **Arquitectura**: MLP con capas Dense
        - **Entrada**: Secuencias temporales 3D
        - **Salida**: Predicciones multi-step
        - **Métricas**: RMSE, MAE por paso
        
        ### 🎯 Ventajas de esta implementación:
        - ✅ Mantiene estructura temporal de los datos
        - ✅ Red neuronal profunda optimizada
        - ✅ Visualizaciones interactivas 3D
        - ✅ Métricas detalladas por paso
        - ✅ Interfaz web intuitiva
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🌌 Predictor de Trayectorias de Asteroides - Desarrollado con TensorFlow y Streamlit</p>
</div>
""", unsafe_allow_html=True)
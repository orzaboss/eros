import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
import base64

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Trayectorias de Asteroides",
    page_icon="🪐",
    layout="wide"
)

# Título principal
st.title("🪐 Predictor de Trayectorias de Asteroides")
st.markdown("### Aplicación de Machine Learning para predecir movimientos orbitales")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración del Modelo")
LOOKBACK = st.sidebar.slider("Ventana de observación (lookback)", 5, 20, 10)
PREDICTION = st.sidebar.slider("Pasos de predicción", 5, 20, 10)
EPOCHS = st.sidebar.slider("Épocas de entrenamiento", 10, 100, 20)

# Funciones del modelo original
@st.cache_data
def extract_datetime(text):
    match = re.search(r'\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+', str(text))
    return match.group(0) if match else None

def load_and_clean(uploaded_file):
    """Carga y limpia archivo desde Streamlit file uploader"""
    try:
        # Leer el archivo subido
        content = uploaded_file.read().decode('utf-8')
        lines = content.split('\n')
        
        # Saltar las primeras 79 líneas como en el código original
        data_lines = lines[79:]
        
        # Procesar líneas
        processed_data = []
        for line in data_lines:
            if line.strip():  # Evitar líneas vacías
                try:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 10:  # Asegurar que tenga suficientes columnas
                        processed_data.append(parts)
                except:
                    continue
        
        if not processed_data:
            return pd.DataFrame()
        
        # Crear DataFrame
        df_raw = pd.DataFrame(processed_data)
        
        # Extraer datetime
        df_raw['datetime_str'] = df_raw[1].apply(extract_datetime)
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime_str'], errors='coerce')
        
        # Extraer columnas numéricas (posiciones X,Y,Z y velocidades VX,VY,VZ)
        df_numeric = df_raw.iloc[:, 4:10].copy()
        df_numeric.columns = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
        df_numeric['datetime'] = df_raw['datetime']
        
        # Convertir a numérico
        for col in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        
        # Limpiar datos
        df_numeric = df_numeric.dropna().reset_index(drop=True)
        return df_numeric
        
    except Exception as e:
        st.error(f"Error procesando archivo: {e}")
        return pd.DataFrame()

def scale_features(df):
    """Escala las características usando MinMaxScaler"""
    scalers = {}
    df_scaled = pd.DataFrame()
    for col in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
        scaler = MinMaxScaler()
        df_scaled[col] = scaler.fit_transform(df[[col]]).flatten()
        scalers[col] = scaler
    df_scaled['datetime'] = df['datetime']
    return df_scaled, scalers

def create_sequences(df, lookback, prediction):
    """Crea secuencias para el modelo"""
    data = df[['X','Y','Z','VX','VY','VZ']].values
    X, y = [], []
    for i in range(len(data) - lookback - prediction + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+prediction])
    return np.array(X), np.array(y)

def create_model(lookback, prediction):
    """Crea el modelo de red neuronal"""
    model = Sequential([
        Flatten(input_shape=(lookback, 6)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(prediction * 6)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

def inverse_transform_data(data_scaled, scalers):
    """Desescala los datos usando los scalers guardados"""
    data_inv = np.empty_like(data_scaled)
    features = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    for i, feat in enumerate(features):
        data_inv[:, :, i] = scalers[feat].inverse_transform(
            data_scaled[:, :, i].reshape(-1, 1)
        ).reshape(-1, data_scaled.shape[1])
    return data_inv

def plot_3d_trajectory(y_true, y_pred, asteroid_name, step):
    """Crea gráfico 3D de trayectorias"""
    fig = go.Figure()
    
    # Trayectoria real
    fig.add_trace(go.Scatter3d(
        x=y_true[:, step, 0],
        y=y_true[:, step, 1],
        z=y_true[:, step, 2],
        mode='lines+markers',
        name='Trayectoria Real',
        line=dict(color='blue', width=4),
        marker=dict(size=3)
    ))
    
    # Trayectoria predicha
    fig.add_trace(go.Scatter3d(
        x=y_pred[:, step, 0],
        y=y_pred[:, step, 1],
        z=y_pred[:, step, 2],
        mode='lines+markers',
        name='Trayectoria Predicha',
        line=dict(color='red', width=4),
        marker=dict(size=3)
    ))
    
    fig.update_layout(
        title=f'{asteroid_name} - Trayectoria 3D (Paso +{step+1})',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600
    )
    
    return fig

def plot_velocities(y_true, y_pred, asteroid_name, step):
    """Crea gráfico de velocidades"""
    features = ['VX', 'VY', 'VZ']
    fig = go.Figure()
    
    for i, feat in enumerate(features):
        # Velocidad real
        fig.add_trace(go.Scatter(
            y=y_true[:, step, i+3],
            mode='lines',
            name=f'Real {feat}',
            line=dict(width=2)
        ))
        
        # Velocidad predicha
        fig.add_trace(go.Scatter(
            y=y_pred[:, step, i+3],
            mode='lines',
            name=f'Predicha {feat}',
            line=dict(dash='dash', width=2)
        ))
    
    fig.update_layout(
        title=f'{asteroid_name} - Velocidades (Paso +{step+1})',
        xaxis_title='Secuencia',
        yaxis_title='Velocidad (km/s)',
        width=800,
        height=400
    )
    
    return fig

# Interfaz principal
st.header("📁 Carga de Datos")

# Área de carga de archivos
uploaded_files = st.file_uploader(
    "Sube archivos de datos de asteroides (.txt)",
    type=['txt'],
    accept_multiple_files=True,
    help="Sube uno o más archivos de datos de asteroides en formato .txt"
)

if uploaded_files:
    st.success(f"Se subieron {len(uploaded_files)} archivos")
    
    # Procesar archivos
    all_data = {}
    all_scalers = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f'Procesando {uploaded_file.name}...')
        
        # Procesar archivo
        df = load_and_clean(uploaded_file)
        
        if not df.empty:
            asteroid_name = uploaded_file.name.replace('.txt', '')
            all_data[asteroid_name] = df
            
            # Mostrar información básica
            st.write(f"**{asteroid_name}**: {len(df)} puntos de datos")
            
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text('¡Procesamiento completado!')
    
    if all_data:
        st.header("🤖 Entrenamiento del Modelo")
        
        if st.button("Entrenar Modelo", type="primary"):
            # Preparar datos para entrenamiento
            X_train_list = []
            y_train_list = []
            
            with st.spinner('Preparando datos y entrenando modelo...'):
                for name, df in all_data.items():
                    df_scaled, scalers = scale_features(df)
                    all_scalers[name] = scalers
                    X_seq, y_seq = create_sequences(df_scaled, LOOKBACK, PREDICTION)
                    if X_seq.size > 0:
                        X_train_list.append(X_seq)
                        y_train_list.append(y_seq)
                
                if X_train_list:
                    X_train = np.concatenate(X_train_list, axis=0)
                    y_train = np.concatenate(y_train_list, axis=0)
                    
                    st.write(f"Dataset de entrenamiento: {X_train.shape[0]} secuencias")
                    
                    # Crear y entrenar modelo
                    model = create_model(LOOKBACK, PREDICTION)
                    y_train_flat = y_train.reshape((y_train.shape[0], PREDICTION * 6))
                    
                    # Entrenar con progress bar
                    progress_placeholder = st.empty()
                    history = model.fit(
                        X_train, y_train_flat,
                        epochs=EPOCHS,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    st.success("¡Modelo entrenado exitosamente!")
                    
                    # Guardar modelo y scalers en session state
                    st.session_state['model'] = model
                    st.session_state['scalers'] = all_scalers
                    st.session_state['data'] = all_data
                    
                    # Mostrar métricas de entrenamiento
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Pérdida Entrenamiento',
                        line=dict(color='blue')
                    ))
                    loss_fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Pérdida Validación',
                        line=dict(color='red')
                    ))
                    loss_fig.update_layout(
                        title='Pérdida durante el Entrenamiento',
                        xaxis_title='Época',
                        yaxis_title='Pérdida (MSE)'
                    )
                    st.plotly_chart(loss_fig, use_container_width=True)

# Sección de predicciones
if 'model' in st.session_state:
    st.header("🔮 Predicciones y Visualizaciones")
    
    # Seleccionar asteroide para evaluar
    asteroid_names = list(st.session_state['data'].keys())
    selected_asteroid = st.selectbox(
        "Selecciona un asteroide para evaluar:",
        asteroid_names
    )
    
    if selected_asteroid:
        col1, col2 = st.columns(2)
        
        with col1:
            step_to_visualize = st.slider(
                "Paso de predicción a visualizar",
                1, PREDICTION, 1
            ) - 1
        
        with col2:
            max_sequences = st.slider(
                "Número máximo de secuencias a mostrar",
                10, 200, 50
            )
        
        if st.button("Generar Predicciones y Gráficos"):
            with st.spinner('Generando predicciones...'):
                # Preparar datos del asteroide seleccionado
                df = st.session_state['data'][selected_asteroid]
                df_scaled, scalers = scale_features(df)
                X_test, y_test = create_sequences(df_scaled, LOOKBACK, PREDICTION)
                
                # Limitar número de secuencias
                if len(X_test) > max_sequences:
                    X_test = X_test[:max_sequences]
                    y_test = y_test[:max_sequences]
                
                # Realizar predicciones
                model = st.session_state['model']
                y_pred_flat = model.predict(X_test)
                y_pred_scaled = y_pred_flat.reshape(y_test.shape)
                
                # Desescalar datos
                y_true = inverse_transform_data(y_test, scalers)
                y_pred = inverse_transform_data(y_pred_scaled, scalers)
                
                # Calcular métricas
                rmse = np.sqrt(mean_squared_error(
                    y_true[:, step_to_visualize, :],
                    y_pred[:, step_to_visualize, :]
                ))
                mae = mean_absolute_error(
                    y_true[:, step_to_visualize, :],
                    y_pred[:, step_to_visualize, :]
                )
                
                # Mostrar métricas
                col1, col2 = st.columns(2)
                col1.metric("RMSE", f"{rmse:.4f}")
                col2.metric("MAE", f"{mae:.4f}")
                
                # Gráfico 3D de trayectorias
                st.subheader("Trayectoria 3D")
                fig_3d = plot_3d_trajectory(y_true, y_pred, selected_asteroid, step_to_visualize)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Gráfico de velocidades
                st.subheader("Velocidades")
                fig_vel = plot_velocities(y_true, y_pred, selected_asteroid, step_to_visualize)
                st.plotly_chart(fig_vel, use_container_width=True)
                
                # Tabla de errores por paso
                st.subheader("Errores por Paso de Predicción")
                error_data = []
                for step in range(PREDICTION):
                    step_rmse = np.sqrt(mean_squared_error(
                        y_true[:, step, :],
                        y_pred[:, step, :]
                    ))
                    step_mae = mean_absolute_error(
                        y_true[:, step, :],
                        y_pred[:, step, :]
                    )
                    error_data.append({
                        'Paso': step + 1,
                        'RMSE': f"{step_rmse:.4f}",
                        'MAE': f"{step_mae:.4f}"
                    })
                
                error_df = pd.DataFrame(error_data)
                st.dataframe(error_df, use_container_width=True)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Información del Modelo")
st.sidebar.markdown("""
- **Arquitectura**: Red Neuronal MLP
- **Entrada**: Secuencias de posición (X,Y,Z) y velocidad (VX,VY,VZ)
- **Salida**: Predicción multi-paso de trayectorias
- **Normalización**: MinMaxScaler por característica
""")

st.sidebar.markdown("### 🚀 Instrucciones")
st.sidebar.markdown("""
1. Sube archivos .txt con datos de asteroides
2. Configura los parámetros del modelo
3. Entrena el modelo con tus datos
4. Genera predicciones y visualizaciones
""")

# Footer
st.markdown("---")
st.markdown("**Predictor de Trayectorias de Asteroides** - Desarrollado con Streamlit y TensorFlow")
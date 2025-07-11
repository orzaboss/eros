import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
import pickle

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Trayectorias de Asteroides",
    page_icon="🌌",
    layout="wide"
)

# Título principal
st.title("🌌 Predictor de Trayectorias de Asteroides")
st.markdown("**Análisis y predicción de órbitas de asteroides usando Machine Learning**")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")
LOOKBACK = st.sidebar.slider("Pasos históricos (LOOKBACK)", 5, 20, 10)
PREDICTION = st.sidebar.slider("Pasos a predecir (PREDICTION)", 5, 20, 10)

# Funciones auxiliares
@st.cache_data
def extract_datetime(text):
    match = re.search(r'\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+', str(text))
    if match:
        return match.group(0)
    else:
        return None

@st.cache_data
def load_and_clean_from_upload(file_content):
    """Procesa archivo subido"""
    try:
        # Leer desde el contenido del archivo
        lines = file_content.decode('utf-8').split('\n')
        
        # Saltar las primeras 79 líneas (header)
        data_lines = lines[79:]
        
        # Procesar líneas
        processed_data = []
        for line in data_lines:
            if line.strip():  # Solo líneas no vacías
                parts = line.split(',')
                if len(parts) >= 10:
                    processed_data.append(parts)
        
        if not processed_data:
            return pd.DataFrame()
        
        # Crear DataFrame
        df_raw = pd.DataFrame(processed_data)
        
        # Extraer datetime
        df_raw['datetime_str'] = df_raw[1].apply(extract_datetime)
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime_str'], errors='coerce')
        
        # Columnas numéricas
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

@st.cache_data
def scale_features(df):
    """Escala las características"""
    scalers = {}
    df_scaled = pd.DataFrame()
    
    for col in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
        scaler = MinMaxScaler()
        df_scaled[col] = scaler.fit_transform(df[[col]]).flatten()
        scalers[col] = scaler
    
    df_scaled['datetime'] = df['datetime']
    return df_scaled, scalers

@st.cache_data
def create_sequences(df, lookback, prediction):
    """Crea secuencias para entrenamiento"""
    data = df[['X','Y','Z','VX','VY','VZ']].values
    X, y = [], []
    
    for i in range(len(data) - lookback - prediction + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+prediction])
    
    return np.array(X), np.array(y)

@st.cache_resource
def create_model(lookback, prediction):
    """Crea el modelo MLP"""
    model = Sequential([
        Flatten(input_shape=(lookback, 6)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(prediction * 6)
    ])
    
    model.compile(optimizer=Adam(), loss='mse')
    return model

def inverse_transform_data(data_scaled, scalers):
    """Desescala los datos"""
    data_inv = np.empty_like(data_scaled)
    features = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    
    for i, feat in enumerate(features):
        data_inv[:, :, i] = scalers[feat].inverse_transform(
            data_scaled[:, :, i].reshape(-1, 1)
        ).reshape(-1, data_scaled.shape[1])
    
    return data_inv

def create_3d_trajectory_plot(y_true, y_pred, asteroid_name, step):
    """Crea gráfico 3D de trayectorias"""
    fig = go.Figure()
    
    # Trayectoria real
    fig.add_trace(go.Scatter3d(
        x=y_true[:, step, 0],
        y=y_true[:, step, 1],
        z=y_true[:, step, 2],
        mode='markers+lines',
        name='Trayectoria Real',
        marker=dict(size=3, color='blue'),
        line=dict(width=2)
    ))
    
    # Trayectoria predicha
    fig.add_trace(go.Scatter3d(
        x=y_pred[:, step, 0],
        y=y_pred[:, step, 1],
        z=y_pred[:, step, 2],
        mode='markers+lines',
        name='Trayectoria Predicha',
        marker=dict(size=3, color='red'),
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'{asteroid_name} - Posiciones 3D (Paso +{step+1})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=600
    )
    
    return fig

def create_velocity_plot(y_true, y_pred, asteroid_name, step):
    """Crea gráfico de velocidades"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Velocidad X', 'Velocidad Y', 'Velocidad Z'),
        vertical_spacing=0.1
    )
    
    velocity_features = ['VX', 'VY', 'VZ']
    colors = ['red', 'green', 'blue']
    
    for i, (feat, color) in enumerate(zip(velocity_features, colors)):
        # Velocidad real
        fig.add_trace(
            go.Scatter(
                y=y_true[:, step, i+3],
                name=f'Real {feat}',
                line=dict(color=color, width=2),
                mode='lines'
            ),
            row=i+1, col=1
        )
        
        # Velocidad predicha
        fig.add_trace(
            go.Scatter(
                y=y_pred[:, step, i+3],
                name=f'Pred {feat}',
                line=dict(color=color, width=2, dash='dash'),
                mode='lines'
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title=f'{asteroid_name} - Velocidades (Paso +{step+1})',
        height=800
    )
    
    return fig

# Interfaz principal
st.header("📁 Carga de Datos")

# Carga de archivos
uploaded_files = st.file_uploader(
    "Sube archivos de asteroides (.txt)",
    type=['txt'],
    accept_multiple_files=True,
    help="Archivos con datos orbitales de asteroides"
)

if uploaded_files:
    # Procesar archivos
    asteroid_data = {}
    
    with st.spinner("Procesando archivos..."):
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            df = load_and_clean_from_upload(file_content)
            
            if not df.empty:
                asteroid_name = uploaded_file.name.replace('.txt', '')
                asteroid_data[asteroid_name] = df
                st.success(f"✅ {asteroid_name}: {len(df)} puntos de datos cargados")
            else:
                st.error(f"❌ Error procesando {uploaded_file.name}")
    
    if asteroid_data:
        st.header("🎯 Entrenamiento del Modelo")
        
        # Preparar datos para entrenamiento
        X_train_list = []
        y_train_list = []
        scalers_global = {}
        
        with st.spinner("Preparando datos para entrenamiento..."):
            for name, df in asteroid_data.items():
                df_scaled, scalers = scale_features(df)
                scalers_global[name] = scalers
                
                X_seq, y_seq = create_sequences(df_scaled, LOOKBACK, PREDICTION)
                
                if X_seq.shape[0] > 0:
                    X_train_list.append(X_seq)
                    y_train_list.append(y_seq)
                
                st.write(f"**{name}**: {X_seq.shape[0]} secuencias creadas")
        
        if X_train_list:
            # Combinar datos
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            y_train_flat = y_train.reshape((y_train.shape[0], PREDICTION * 6))
            
            st.write(f"**Dataset total**: {X_train.shape[0]} secuencias")
            
            # Entrenar modelo
            if st.button("🚀 Entrenar Modelo"):
                with st.spinner("Entrenando modelo..."):
                    model = create_model(LOOKBACK, PREDICTION)
                    
                    # Crear contenedor para progreso
                    progress_bar = st.progress(0)
                    epochs = 20
                    
                    # Entrenar por lotes para mostrar progreso
                    for epoch in range(epochs):
                        model.fit(X_train, y_train_flat, 
                                epochs=1, batch_size=32, 
                                validation_split=0.2, verbose=0)
                        progress_bar.progress((epoch + 1) / epochs)
                    
                    st.success("🎉 Modelo entrenado exitosamente!")
                    
                    # Guardar modelo en session state
                    st.session_state['model'] = model
                    st.session_state['scalers'] = scalers_global
                    st.session_state['asteroid_data'] = asteroid_data
        
        # Evaluación y visualización
        if 'model' in st.session_state:
            st.header("📊 Evaluación y Visualización")
            
            # Seleccionar asteroide para evaluar
            selected_asteroid = st.selectbox(
                "Selecciona un asteroide para evaluar:",
                list(asteroid_data.keys())
            )
            
            if selected_asteroid:
                df_selected = asteroid_data[selected_asteroid]
                df_scaled, scalers = scale_features(df_selected)
                X_test, y_test = create_sequences(df_scaled, LOOKBACK, PREDICTION)
                
                if X_test.shape[0] > 0:
                    # Hacer predicciones
                    model = st.session_state['model']
                    y_pred_flat = model.predict(X_test)
                    y_pred_scaled = y_pred_flat.reshape(y_test.shape)
                    
                    # Desescalar
                    y_true = inverse_transform_data(y_test, scalers)
                    y_pred = inverse_transform_data(y_pred_scaled, scalers)
                    
                    # Métricas
                    st.subheader("📈 Métricas de Error")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Error por paso de predicción:**")
                        for step in range(min(5, PREDICTION)):  # Mostrar solo primeros 5 pasos
                            rmse = np.sqrt(mean_squared_error(y_true[:, step, :], y_pred[:, step, :]))
                            mae = mean_absolute_error(y_true[:, step, :], y_pred[:, step, :])
                            st.write(f"Paso +{step+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
                    
                    with col2:
                        # Gráfico de error
                        steps = list(range(1, PREDICTION + 1))
                        rmse_values = []
                        mae_values = []
                        
                        for step in range(PREDICTION):
                            rmse = np.sqrt(mean_squared_error(y_true[:, step, :], y_pred[:, step, :]))
                            mae = mean_absolute_error(y_true[:, step, :], y_pred[:, step, :])
                            rmse_values.append(rmse)
                            mae_values.append(mae)
                        
                        fig_error = go.Figure()
                        fig_error.add_trace(go.Scatter(x=steps, y=rmse_values, name='RMSE', mode='lines+markers'))
                        fig_error.add_trace(go.Scatter(x=steps, y=mae_values, name='MAE', mode='lines+markers'))
                        fig_error.update_layout(title='Error por Paso de Predicción', 
                                              xaxis_title='Paso', yaxis_title='Error')
                        st.plotly_chart(fig_error, use_container_width=True)
                    
                    # Visualizaciones
                    st.subheader("🌌 Visualizaciones")
                    
                    # Selector de paso
                    step_to_plot = st.selectbox(
                        "Selecciona el paso a visualizar:",
                        list(range(1, PREDICTION + 1)),
                        index=0
                    ) - 1
                    
                    # Gráfico 3D
                    fig_3d = create_3d_trajectory_plot(y_true, y_pred, selected_asteroid, step_to_plot)
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Gráfico de velocidades
                    fig_vel = create_velocity_plot(y_true, y_pred, selected_asteroid, step_to_plot)
                    st.plotly_chart(fig_vel, use_container_width=True)
                    
                    # Información del dataset
                    st.subheader("ℹ️ Información del Dataset")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Puntos de datos", len(df_selected))
                    with col2:
                        st.metric("Secuencias creadas", X_test.shape[0])
                    with col3:
                        st.metric("Características", 6)
                    
                    # Estadísticas del asteroide
                    st.subheader("📋 Estadísticas del Asteroide")
                    st.dataframe(df_selected[['X', 'Y', 'Z', 'VX', 'VY', 'VZ']].describe())
                    
                else:
                    st.error("No hay suficientes datos para crear secuencias de prueba")
    
    # Información y ayuda
    with st.expander("❓ Ayuda"):
        st.markdown("""
        ### Cómo usar esta aplicación:
        
        1. **Carga archivos**: Sube archivos .txt con datos orbitales de asteroides
        2. **Configura parámetros**: Ajusta LOOKBACK y PREDICTION en el sidebar
        3. **Entrena modelo**: Haz clic en "Entrenar Modelo" para crear el modelo predictivo
        4. **Evalúa resultados**: Selecciona un asteroide y visualiza las predicciones
        
        ### Formato de archivos:
        - Los archivos deben tener al menos 79 líneas de header
        - Las columnas deben incluir: X, Y, Z, VX, VY, VZ
        - Los datos deben estar separados por comas
        
        ### Parámetros:
        - **LOOKBACK**: Número de pasos históricos para hacer predicciones
        - **PREDICTION**: Número de pasos futuros a predecir
        """)

else:
    st.info("👆 Sube archivos de asteroides para comenzar el análisis")
    
    # Ejemplo de formato de archivo
    st.subheader("📄 Formato de archivo esperado")
    st.code("""
    # Primeras 79 líneas son header (se ignoran)
    # Después vienen los datos con formato:
    JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ, ...
    2451545.0, 2000-Jan-01 12:00:00.000, 1.234, 2.345, 3.456, 0.123, 0.234, 0.345, ...
    """)
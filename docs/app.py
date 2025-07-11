import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io

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
PREDICTION = st.sidebar.slider("Pasos a predecir (PREDICTION)", 1, 10, 5)
algorithm = st.sidebar.selectbox("Algoritmo", ["MLP Neural Network", "Random Forest"])

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
        lines = file_content.decode('utf-8').split('\n')
        data_lines = lines[79:]
        
        processed_data = []
        for line in data_lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 10:
                    processed_data.append(parts)
        
        if not processed_data:
            return pd.DataFrame()
        
        df_raw = pd.DataFrame(processed_data)
        df_raw['datetime_str'] = df_raw[1].apply(extract_datetime)
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime_str'], errors='coerce')
        
        df_numeric = df_raw.iloc[:, 4:10].copy()
        df_numeric.columns = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
        df_numeric['datetime'] = df_raw['datetime']
        
        for col in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        
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

def create_sequences_for_sklearn(df, lookback, prediction):
    """Crea secuencias para sklearn (2D)"""
    data = df[['X','Y','Z','VX','VY','VZ']].values
    X, y = [], []
    
    for i in range(len(data) - lookback - prediction + 1):
        X.append(data[i:i+lookback].flatten())  # Aplanar para sklearn
        y.append(data[i+lookback:i+lookback+prediction].flatten())  # Aplanar salida
    
    return np.array(X), np.array(y)

def create_model(algorithm, lookback, prediction, n_features=6):
    """Crea modelo usando sklearn"""
    if algorithm == "MLP Neural Network":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            random_state=42
        )
    else:  # Random Forest
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

def inverse_transform_prediction(pred_flat, scalers, prediction, n_features=6):
    """Desescala predicciones planas"""
    pred_reshaped = pred_flat.reshape(-1, prediction, n_features)
    pred_descaled = np.empty_like(pred_reshaped)
    
    features = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    for i, feat in enumerate(features):
        for step in range(prediction):
            pred_descaled[:, step, i] = scalers[feat].inverse_transform(
                pred_reshaped[:, step, i].reshape(-1, 1)
            ).flatten()
    
    return pred_descaled

def create_3d_trajectory_plot(y_true, y_pred, asteroid_name, step):
    """Crea gráfico 3D de trayectorias"""
    fig = go.Figure()
    
    # Limitar número de puntos para mejor visualización
    max_points = min(100, len(y_true))
    idx = np.linspace(0, len(y_true)-1, max_points, dtype=int)
    
    fig.add_trace(go.Scatter3d(
        x=y_true[idx, step, 0],
        y=y_true[idx, step, 1],
        z=y_true[idx, step, 2],
        mode='markers+lines',
        name='Trayectoria Real',
        marker=dict(size=3, color='blue'),
        line=dict(width=3)
    ))
    
    fig.add_trace(go.Scatter3d(
        x=y_pred[idx, step, 0],
        y=y_pred[idx, step, 1],
        z=y_pred[idx, step, 2],
        mode='markers+lines',
        name='Trayectoria Predicha',
        marker=dict(size=3, color='red'),
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title=f'{asteroid_name} - Posiciones 3D (Paso +{step+1})',
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)'
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
    
    # Limitar puntos para mejor visualización
    max_points = min(50, len(y_true))
    idx = np.linspace(0, len(y_true)-1, max_points, dtype=int)
    
    for i, (feat, color) in enumerate(zip(velocity_features, colors)):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(idx))),
                y=y_true[idx, step, i+3],
                name=f'Real {feat}',
                line=dict(color=color, width=2),
                mode='lines+markers'
            ),
            row=i+1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(idx))),
                y=y_pred[idx, step, i+3],
                name=f'Pred {feat}',
                line=dict(color=color, width=2, dash='dash'),
                mode='lines+markers'
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

uploaded_files = st.file_uploader(
    "Sube archivos de asteroides (.txt)",
    type=['txt'],
    accept_multiple_files=True,
    help="Archivos con datos orbitales de asteroides"
)

if uploaded_files:
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
                
                X_seq, y_seq = create_sequences_for_sklearn(df_scaled, LOOKBACK, PREDICTION)
                
                if X_seq.shape[0] > 0:
                    X_train_list.append(X_seq)
                    y_train_list.append(y_seq)
                
                st.write(f"**{name}**: {X_seq.shape[0]} secuencias creadas")
        
        if X_train_list:
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            
            st.write(f"**Dataset total**: {X_train.shape[0]} secuencias")
            st.write(f"**Forma de entrada**: {X_train.shape}")
            st.write(f"**Forma de salida**: {y_train.shape}")
            
            # Entrenar modelo
            if st.button("🚀 Entrenar Modelo"):
                with st.spinner(f"Entrenando modelo {algorithm}..."):
                    model = create_model(algorithm, LOOKBACK, PREDICTION)
                    
                    # Entrenar
                    model.fit(X_train, y_train)
                    
                    st.success(f"🎉 Modelo {algorithm} entrenado exitosamente!")
                    
                    # Calcular score en entrenamiento
                    train_score = model.score(X_train, y_train)
                    st.write(f"**Score R² en entrenamiento**: {train_score:.4f}")
                    
                    # Guardar modelo en session state
                    st.session_state['model'] = model
                    st.session_state['scalers'] = scalers_global
                    st.session_state['asteroid_data'] = asteroid_data
                    st.session_state['algorithm'] = algorithm
        
        # Evaluación y visualización
        if 'model' in st.session_state:
            st.header("📊 Evaluación y Visualización")
            
            selected_asteroid = st.selectbox(
                "Selecciona un asteroide para evaluar:",
                list(asteroid_data.keys())
            )
            
            if selected_asteroid:
                df_selected = asteroid_data[selected_asteroid]
                df_scaled, scalers = scale_features(df_selected)
                X_test, y_test = create_sequences_for_sklearn(df_scaled, LOOKBACK, PREDICTION)
                
                if X_test.shape[0] > 0:
                    model = st.session_state['model']
                    y_pred_flat = model.predict(X_test)
                    
                    # Reshape para visualización
                    y_test_reshaped = y_test.reshape(-1, PREDICTION, 6)
                    y_pred_reshaped = y_pred_flat.reshape(-1, PREDICTION, 6)
                    
                    # Desescalar
                    y_true = inverse_transform_prediction(y_test, scalers, PREDICTION)
                    y_pred = inverse_transform_prediction(y_pred_flat, scalers, PREDICTION)
                    
                    # Métricas
                    st.subheader("📈 Métricas de Error")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Error por paso de predicción:**")
                        for step in range(PREDICTION):
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
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Puntos de datos", len(df_selected))
                    with col2:
                        st.metric("Secuencias creadas", X_test.shape[0])
                    with col3:
                        st.metric("Algoritmo", st.session_state['algorithm'])
                    with col4:
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
        3. **Selecciona algoritmo**: Elige entre MLP Neural Network o Random Forest
        4. **Entrena modelo**: Haz clic en "Entrenar Modelo" para crear el modelo predictivo
        5. **Evalúa resultados**: Selecciona un asteroide y visualiza las predicciones
        
        ### Formato de archivos:
        - Los archivos deben tener al menos 79 líneas de header
        - Las columnas deben incluir: X, Y, Z, VX, VY, VZ
        - Los datos deben estar separados por comas
        
        ### Algoritmos disponibles:
        - **MLP Neural Network**: Red neuronal multicapa (sklearn)
        - **Random Forest**: Ensemble de árboles de decisión
        """)

else:
    st.info("👆 Sube archivos de asteroides para comenzar el análisis")
    
    # Datos de ejemplo
    st.subheader("📊 Ejemplo de datos esperados")
    
    # Crear datos de ejemplo
    example_data = {
        'X': [1.234, 1.235, 1.236],
        'Y': [2.345, 2.346, 2.347],
        'Z': [3.456, 3.457, 3.458],
        'VX': [0.123, 0.124, 0.125],
        'VY': [0.234, 0.235, 0.236],
        'VZ': [0.345, 0.346, 0.347]
    }
    
    st.dataframe(pd.DataFrame(example_data))
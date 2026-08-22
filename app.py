import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from pathlib import Path

# Carpeta donde vive app.py (para localizar recursos como la fotografía)
_CARPETA_APP = Path(__file__).resolve().parent
# Pipeline de inferencia y descarga de datos (mismos módulos que el entrenamiento)
from Prediccion import PredictorVolatilidad
from Caracteristicas import descargar_datos
from Benchmarks import predecir_egarch, predecir_garch, predecir_har_rv

# ==========================================
# 1. Configuración inicial de la página
# ==========================================
st.set_page_config(
    page_title="Volatitlán",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. Carga del Modelo en Caché (Optimización)
# ==========================================
@st.cache_resource
def cargar_predictor():
    return PredictorVolatilidad()

try:
    predictor = cargar_predictor()
    modelo_cargado = True
except Exception as e:
    modelo_cargado = False
    error_msg = e

# ==========================================
# 3. Inyección de CSS personalizado 
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Cinzel:wght@400;600;700&family=Montserrat:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif !important;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #F4F6F9; }

    .top-header {
        background: linear-gradient(135deg, #0F253F 0%, #1A365D 100%);
        color: white;
        padding: 1.8rem 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: -3.5rem;
        margin-bottom: 2rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-titles { display: flex; flex-direction: column; }
    .title-text { font-size: 1.8rem; font-weight: 700; margin: 0; letter-spacing: 2px; text-transform: uppercase; }
    .subtitle-text { font-size: 1rem; font-weight: 300; margin-top: 5px; color: #A0B2C6; }
    .author-signature { font-family: 'Great Vibes', cursive !important; font-size: 2.5rem; margin: 0; color: #E2E8F0; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #0F253F !important; border-radius: 8px 8px 0px 0px !important; padding: 12px 30px !important; border: none !important; transition: all 0.3s ease; }
    .stTabs [data-baseweb="tab"] p { color: #cbd5e1 !important; font-size: 1.05rem !important; font-weight: 400 !important; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #1A365D !important; transform: translateY(-2px); }
    .stTabs [aria-selected="true"] { background-color: #0F253F !important; border-bottom: 3px solid #3B82F6 !important; }
    .stTabs [aria-selected="true"] p { color: #ffffff !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* El panel de la pestaña ES la caja azul marino: así las métricas,
       gráficas y botones de Streamlit quedan DENTRO del fondo oscuro */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0F253F;
        border-radius: 0px 8px 8px 8px;
        padding: 3rem !important;
        min-height: 60vh;
        color: white;
        box-shadow: 0 10px 25px rgba(15, 37, 63, 0.15);
    }
    .stTabs [data-baseweb="tab-panel"] h2 {
        color: white;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-panel"] hr { border-color: rgba(255,255,255,0.1); }
    
    /* Adaptar las métricas de Streamlit al fondo oscuro */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(59,130,246,0.5);
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] { color: #3B82F6 !important; font-weight: 700 !important;}
    [data-testid="stMetricLabel"] { color: #A0B2C6 !important; font-size: 1.05rem !important;}

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #1A365D 0%, #3B82F6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(59,130,246,0.25);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 18px rgba(59,130,246,0.45);
        transform: translateY(-2px);
    }

    /* Subtítulos de sección dentro de la caja oscura */
    .section-title {
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0 0 0.3rem 0;
    }
    .section-caption { color: #A0B2C6; font-size: 0.95rem; margin-bottom: 1rem; }

    /* Selector de tiempo (radio horizontal) sobre fondo oscuro */
    .stRadio [data-testid="stWidgetLabel"] p { color: #A0B2C6 !important; }
    .stRadio label p { color: #E2E8F0 !important; font-weight: 500; }

    /* Selector de semana (selectbox) sobre fondo oscuro */
    .stSelectbox [data-testid="stWidgetLabel"] p { color: #A0B2C6 !important; }

    /* ---------- Pestaña: Acerca del Autor ---------- */
    .autor-nombre {
        font-family: 'Cinzel', serif !important;
        font-size: 2.6rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 2px;
        margin: 0 0 0.2rem 0;
        line-height: 1.15;
    }
    .autor-rol {
        color: #3B82F6;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .autor-bio {
        color: #E2E8F0;
        font-size: 1.65rem;
        font-weight: 300;
        line-height: 1.6;
        max-width: 46ch;
        margin-top: 1.2rem;
    }
    .autor-foto img {
        border-radius: 12px;
        border: 2px solid rgba(59,130,246,0.35);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .contacto-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-top: 1.4rem;
    }
    .contacto-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 0.55rem 1.1rem;
        color: #E2E8F0 !important;
        text-decoration: none !important;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.25s ease;
    }
    .contacto-chip:hover {
        background: rgba(59,130,246,0.18);
        border-color: rgba(59,130,246,0.6);
        transform: translateY(-2px);
    }
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #3B82F6;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        height: 100%;
    }
    .info-card h4 {
        color: #FFFFFF; font-size: 1rem; font-weight: 600;
        margin: 0 0 0.6rem 0; letter-spacing: 0.5px;
    }
    .info-card p, .info-card li { color: #A0B2C6; font-size: 0.93rem; line-height: 1.65; margin: 0; }
    .info-card ul { margin: 0; padding-left: 1.1rem; }

    /* Distintivo del mejor modelo: la flecha del delta sobra cuando el texto
       no es una variación numérica */
    .st-key-metrica_mejor [data-testid="stMetricDelta"] svg { display: none; }
    .st-key-metrica_mejor [data-testid="stMetricDelta"] { font-weight: 600; }

    /* ---------- Tabla de fórmulas (Selección de modelos) ---------- */
    .st-key-tabla_formulas {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px;
        padding: 0.4rem 1.4rem 1rem 1.4rem !important;
    }
    /* KaTeX hereda el color del contenedor; se fija para que no dependa
       del tema activo de Streamlit */
    .st-key-tabla_formulas .katex { color: #E2E8F0 !important; font-size: 1.05rem; }
    .st-key-tabla_formulas [data-testid="stElementContainer"] { margin-bottom: 0; }
    .tabla-encabezado {
        color: #3B82F6; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 1.4px; text-transform: uppercase;
        margin: 0.2rem 0 0.1rem 0;
    }
    .formula-sigla { color: #FFFFFF; font-size: 1.05rem; font-weight: 600; margin: 0; }
    .formula-nombre { color: #A0B2C6; font-size: 0.82rem; line-height: 1.4; margin: 0.15rem 0 0 0; }
    .formula-lectura { color: #A0B2C6; font-size: 0.88rem; line-height: 1.6; margin: 0; }
    .fila-sep { border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.55rem 0; }

    .footer-signature { font-family: 'Great Vibes', cursive !important; font-size: 2.2rem; text-align: right; color: #64748B; margin-top: 2rem; padding-right: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3.4 Descarga única de datos de mercado
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def datos_mercado():
    """
    ÚNICA descarga de mercado de la app: histórico completo (desde 1990)
    compartido por todas las pestañas.

    Antes cada sección (backtest, dashboard, comparativa de modelos)
    descargaba por su cuenta: al abrir la app se disparaban hasta seis
    peticiones casi simultáneas a Yahoo, cuyo límite de tasa las rechazaba y
    la app caía al respaldo local aunque Yahoo estuviera disponible — y ese
    resultado degradado quedaba cacheado una hora.
    """
    return descargar_datos()


# ==========================================
# 3.5 Evaluación histórica del pronóstico (Backtest Out-of-Sample)
# ==========================================
def _metricas(real, pred):
    """
    Métricas de pronóstico sobre niveles de RV.

    Se incluye MdAPE y QLIKE además del MAPE: en volatilidad el MAPE está
    dominado por los periodos de calma (denominador pequeño), mientras que
    QLIKE es la pérdida estándar de la literatura y penaliza correctamente
    subestimar la volatilidad en los episodios de estrés.
    """
    real = np.asarray(real, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-8, None)
    err = real - pred
    ape = np.abs(err) / np.abs(real)
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(ape) * 100),
        "MdAPE": float(np.median(ape) * 100),
        "QLIKE": float(np.mean(np.log(pred ** 2) + (real ** 2) / (pred ** 2))),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def evaluar_desempeno_historico(anios=2):
    """
    Reconstruye las predicciones del modelo sobre los últimos `anios` años y
    las compara contra la volatilidad realizada observada y contra el baseline
    de persistencia ("la próxima semana será como la pasada").
    """
    completo = datos_mercado()
    # Buffer amplio: la RV a 66 días y las medias móviles de 22 consumen ~90
    # sesiones de calentamiento antes de la primera fila utilizable.
    inicio = completo.index.max() - pd.Timedelta(days=int(anios * 365) + 320)
    df = completo.loc[completo.index >= inicio]
    feats = predictor.preparar(df)
    X_esc, harx = predictor.matrices(feats)

    # Volatilidad realizada futura observada: RV(t+1..t+5)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rv_5d = np.sqrt((log_ret ** 2).rolling(window=5).sum())
    y = rv_5d.shift(-5).reindex(feats.index).to_numpy()

    # Se excluyen los últimos días: su futuro todavía no se observa
    filas = [i for i in range(predictor.ventana, len(feats)) if not np.isnan(y[i])]
    if not filas:
        raise ValueError("No hay suficientes datos para evaluar el desempeño.")

    pred, filas_ok = predictor.predecir_en_filas(X_esc, harx, filas)
    reales = y[filas_ok]
    naive = feats["RV_5d"].to_numpy()[filas_ok]

    resultado = pd.DataFrame(
        {"Real": reales, "Predicción": pred, "Persistencia": naive},
        index=feats.index[filas_ok],
    )
    metricas = {
        "modelo": _metricas(reales, pred),
        "persistencia": _metricas(reales, naive),
    }
    return resultado, metricas


# ==========================================
# 3.6 Datos y utilidades del Dashboard
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos_dashboard():
    """
    Descarga ~6 años del S&P 500 junto al VIX y calcula los indicadores.

    El VIX es imprescindible: forma parte de las variables del modelo, así que
    este frame alimenta tanto las gráficas del dashboard como el pronóstico.
    Los indicadores técnicos (RSI, MACD, ATR, Momentum) ya no son entradas del
    modelo —fueron sustituidos por medidas de asimetría— pero se conservan
    como lectura visual del mercado en la pestaña de dashboard.
    """
    completo = datos_mercado()
    df = completo.loc[completo.index >= completo.index.max() - pd.Timedelta(days=6 * 365)]

    d = df.copy()
    d["Log_Ret"] = np.log(d["Close"] / d["Close"].shift(1))
    d["Vol_5d"] = d["Log_Ret"].rolling(5).std()
    d["Vol_20d"] = d["Log_Ret"].rolling(20).std()
    d["Vol_60d"] = d["Log_Ret"].rolling(60).std()
    d["RSI"] = ta.momentum.RSIIndicator(d["Close"], window=14).rsi()
    macd = ta.trend.MACD(d["Close"])
    d["MACD"] = macd.macd()
    d["MACD_Senal"] = macd.macd_signal()  # solo para lectura visual del MACD
    d["ATR"] = ta.volatility.AverageTrueRange(d["High"], d["Low"], d["Close"], window=14).average_true_range()
    d["Momento"] = ta.momentum.ROCIndicator(d["Close"], window=10).roc()
    d["Vol_Pct_Change"] = d["Volume"].pct_change()
    limpio = d.replace([np.inf, -np.inf], np.nan).dropna()
    # `attrs` se propaga en la mayoría de operaciones de pandas, pero se fija
    # de forma explícita para que la interfaz pueda avisar del origen.
    limpio.attrs["fuente"] = df.attrs.get("fuente", "yahoo")
    return limpio


def estilo_grafica(fig, titulo, altura=320, leyenda=True):
    """
    Aplica el tema azul marino consistente a cualquier figura Plotly.

    El título se ancla al contenedor (`yref="container"`) y la leyenda se
    coloca por debajo, dentro del margen superior. Con el título en posición
    automática ambos caían en la misma banda de 45 px y se solapaban, que era
    la razón por la que ninguno de los dos se leía. El margen superior de
    88 px reserva espacio para las dos filas sin tocar la altura del gráfico.
    """
    fig.update_layout(
        title=dict(
            text=titulo,
            x=0, xanchor="left",
            yref="container", y=1.0, yanchor="top",
            pad=dict(t=14, l=6),
            font=dict(color="#FFFFFF", size=16),
        ),
        height=altura,
        margin=dict(l=10, r=10, t=88 if leyenda else 58, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        font=dict(color="white", size=12),
        hovermode="x unified",
        showlegend=leyenda,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.015, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0", size=12),
            itemsizing="constant",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


# ==========================================
# 3.65 Pronóstico vigente (se calcula al abrir la app)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def pronostico_actual():
    """
    RV esperada para los próximos 5 días hábiles a partir del último cierre.

    Equivale a `predictor.predecir_futuro()`, pero reutiliza el frame ya
    cacheado en vez de lanzar su propia descarga de 400 días: la pestaña se
    abre con el pronóstico ya calculado, sin pedirle nada más a Yahoo.
    """
    df = datos_mercado()
    feats = predictor.preparar(df)
    if len(feats) < predictor.ventana:
        raise ValueError(
            f"Datos insuficientes: {len(feats)} filas limpias para una "
            f"ventana de {predictor.ventana}."
        )
    X_esc, harx = predictor.matrices(feats)
    niveles, _ = predictor.predecir_en_filas(X_esc, harx, [len(feats) - 1])
    return float(niveles[0]), feats.index[-1]


# ==========================================
# 3.7 Pronóstico semanal vs. realidad
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def historial_semanal(n_semanas=26):
    """
    Serie semanal de pronósticos vs. realidad. Para cada semana, la predicción
    se genera con la información disponible ANTES de su lunes (cierre del
    viernes previo) y se compara con la RV realizada de esa misma semana.
    """
    df = cargar_datos_dashboard()
    feats = predictor.preparar(df)
    X_esc, harx = predictor.matrices(feats)
    idx = feats.index

    ultimo = df.index.max()
    lunes_actual = (ultimo - pd.Timedelta(days=int(ultimo.weekday()))).normalize()
    lunes_lista = [lunes_actual - pd.Timedelta(weeks=k) for k in range(n_semanas - 1, -1, -1)]

    filas, meta = [], {}
    for lunes in lunes_lista:
        pos = int(idx.searchsorted(lunes))  # primera sesión de la semana
        if pos < predictor.ventana or pos >= len(idx):
            continue  # sin historial suficiente para formar la ventana

        sem = df[(df.index >= lunes) & (df.index < lunes + pd.Timedelta(days=7))].head(5)
        filas.append(pos)
        meta[pos] = {
            "Semana": lunes,
            "Real": float(np.sqrt((sem["Log_Ret"] ** 2).sum())) if len(sem) else np.nan,
            "Sesiones": len(sem),
            # Baseline de persistencia: la RV de la semana previa al pronóstico
            "Persistencia": float(feats["RV_5d"].to_numpy()[pos]),
        }

    if not filas:
        raise ValueError("No hay historial suficiente para construir la serie semanal.")

    pred, filas_ok = predictor.predecir_en_filas(X_esc, harx, filas)
    res = pd.DataFrame([meta[p] for p in filas_ok]).set_index("Semana")
    res["Pronostico"] = pred
    res["Completa"] = res["Sesiones"] >= 5
    return res


# ==========================================
# 3.8 Comparativa de modelos (Selección de modelos)
# ==========================================
COLORES_MODELOS = {
    "LSTM": "#3B82F6",
    "HAR-RV": "#34D399",
    "GARCH(1,1)": "#FBBF24",
    "EGARCH": "#C084FC",
}

# Definición formal de cada métrica. Las fórmulas reproducen literalmente lo
# que calcula `_metricas`: si esa función cambia, estas expresiones deben
# cambiar con ella o la pestaña estaría documentando algo que no se ejecuta.
FORMULAS_METRICAS = [
    (
        "MAPE",
        "Error porcentual absoluto medio",
        r"\text{MAPE}=\frac{100}{n}\sum_{t=1}^{n}"
        r"\left|\frac{RV_t-\widehat{RV}_t}{RV_t}\right|",
        "Error relativo promedio, en porcentaje. Es comparable entre modelos y "
        "periodos, pero al dividir entre la RV observada se dispara en los tramos "
        "de calma, donde el denominador es pequeño.",
    ),
    (
        "MdAPE",
        "Mediana del error porcentual",
        r"\text{MdAPE}=100\cdot\operatorname{mediana}_{t}"
        r"\left|\frac{RV_t-\widehat{RV}_t}{RV_t}\right|",
        "La versión robusta del MAPE: describe el error del día típico y no se "
        "deja arrastrar por unos pocos episodios extremos.",
    ),
    (
        "RMSE",
        "Raíz del error cuadrático medio",
        r"\text{RMSE}=\sqrt{\frac{1}{n}\sum_{t=1}^{n}"
        r"\left(RV_t-\widehat{RV}_t\right)^{2}}",
        "Error absoluto en unidades de volatilidad. Al elevar al cuadrado castiga "
        "con dureza los fallos grandes, típicamente los de los shocks.",
    ),
    (
        "MAE",
        "Error absoluto medio",
        r"\text{MAE}=\frac{1}{n}\sum_{t=1}^{n}"
        r"\left|RV_t-\widehat{RV}_t\right|",
        "Error absoluto promedio, sin elevar al cuadrado: más estable que el RMSE "
        "y más fácil de leer, pues comparte las unidades de la RV.",
    ),
    (
        "QLIKE",
        "Pérdida cuasi-verosímil",
        r"\text{QLIKE}=\frac{1}{n}\sum_{t=1}^{n}"
        r"\left[\ln\widehat{RV}_t^{\,2}+\frac{RV_t^{2}}{\widehat{RV}_t^{2}}\right]",
        "La pérdida estándar de la literatura de volatilidad. Es asimétrica a "
        "propósito: penaliza mucho más subestimar la volatilidad que sobreestimarla, "
        "que es el error costoso en la práctica.",
    ),
]


@st.cache_data(ttl=3600, show_spinner=False)
def comparativa_modelos(anios=2):
    """
    Evalúa la LSTM y los tres benchmarks econométricos sobre el mismo periodo
    out-of-sample y el mismo objetivo: RV(t+1..t+5).

    Se descarga el histórico completo (desde 1990) porque los GARCH y el
    HAR-RV se estiman con TODA la muestra anterior al corte; evaluar un GARCH
    ajustado con dos años de datos sería injusto con el benchmark.
    """
    df = datos_mercado()
    feats = predictor.preparar(df)
    corte = feats.index.max() - pd.Timedelta(days=int(anios * 365))

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rv_5d = np.sqrt((log_ret ** 2).rolling(window=5).sum())
    y = rv_5d.shift(-5).reindex(feats.index).to_numpy()

    filas = [i for i in range(predictor.ventana, len(feats))
             if feats.index[i] > corte and not np.isnan(y[i])]
    if not filas:
        raise ValueError("No hay suficientes datos para construir la comparativa.")

    X_esc, harx = predictor.matrices(feats)
    pred_lstm, filas_ok = predictor.predecir_en_filas(X_esc, harx, filas)
    idx_eval = feats.index[filas_ok]

    # Benchmarks: parámetros congelados en el corte, predicción posterior
    har = predecir_har_rv(feats, corte)
    garch = predecir_garch(log_ret, corte)
    egarch = predecir_egarch(log_ret, corte)

    res = pd.DataFrame(
        {
            "Real": pd.Series(y, index=feats.index).loc[idx_eval],
            "LSTM": pred_lstm,
            "HAR-RV": har.reindex(idx_eval),
            "GARCH(1,1)": garch.reindex(idx_eval),
            "EGARCH": egarch.reindex(idx_eval),
        },
        index=idx_eval,
    ).dropna()

    metricas = {m: _metricas(res["Real"], res[m]) for m in COLORES_MODELOS}
    return res, metricas


# ==========================================
# 4. Construcción del Header
# ==========================================
st.markdown("""
    <div class="top-header">
        <div class="header-titles">
            <div class="title-text">Volatitlán</div>
            <div class="subtitle-text">Pronóstico de volatilidades</div>
        </div>
        <div class="author-signature">Por Materanda</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 5. Creación de las Pestañas
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["Pronóstico", "Dashboard", "Selección de modelos", "Acerca del Autor"])

# --- CONTENIDO PESTAÑA 1: PRONÓSTICO Y MÉTRICAS ---
with tab1:
    st.markdown("""
        <h2>Pronóstico LSTM</h2>
        <p style="color: #A0B2C6; font-size: 1.1rem;">Análisis y predicción de la volatilidad del índice S&P 500 utilizando redes neuronales recurrentes.</p>
    """, unsafe_allow_html=True)
    
    if not modelo_cargado:
        st.error(f"No se pudo cargar el modelo. Verifica que 'modelo_lstm_sp500_volatilidad.keras' y los escaladores existan. Detalle: {error_msg}")
    else:
        # Aviso de frescura: Yahoo Finance bloquea peticiones desde IPs de
        # centros de datos, así que en la nube la descarga puede caer al
        # respaldo local. Mostrar pronósticos con datos viejos sin avisar
        # sería peor que no mostrarlos.
        try:
            _datos = cargar_datos_dashboard()
            _ultimo = _datos.index.max()
            _dias = (pd.Timestamp.today().normalize() - _ultimo.normalize()).days
            _fuente = _datos.attrs.get("fuente", "yahoo")
            if _fuente == "respaldo":
                st.warning(
                    f"Yahoo Finance no respondió; se está usando la copia local del "
                    f"histórico, con datos hasta el {_ultimo:%d/%m/%Y} ({_dias} días). "
                    f"Los pronósticos no incorporan las sesiones más recientes."
                )
            elif _dias > 5:
                st.info(
                    f"Último dato de mercado disponible: {_ultimo:%d/%m/%Y} "
                    f"({_dias} días de antigüedad)."
                )
        except Exception:
            pass  # el aviso es accesorio: nunca debe tumbar la página

        # Fila 1: Pronóstico vigente. Se calcula al abrir la pestaña; antes
        # exigía pulsar un botón, que solo añadía un paso para ver el dato
        # principal de la aplicación.
        st.markdown("<p class='section-title'>Pronóstico Actualizado</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-caption'>Predicción de volatilidad para los próximos 5 días hábiles, generada con los datos de mercado más recientes disponibles.</p>", unsafe_allow_html=True)

        try:
            with st.spinner(f"Procesando tensores de {predictor.ventana} días..."):
                rv_futura, fecha_base = pronostico_actual()
                df_plot = datos_mercado().tail(60)  # contexto visual, más amplio que la ventana

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric(
                    label="Volatilidad Realizada (Próximos 5 días)",
                    value=f"{rv_futura:.4f}",
                )
                # Anualizar la RV semanal la vuelve comparable con el VIX y con
                # cualquier cifra de volatilidad que se cite en el mercado.
                st.metric(
                    label="Equivalente anualizado",
                    value=f"{rv_futura * np.sqrt(252.0 / predictor.horizonte) * 100:.1f}%",
                )
                st.markdown(
                    f"<p class='section-caption'>Calculado con la información hasta el cierre del "
                    f"{fecha_base:%d/%m/%Y}. Es la raíz cuadrada de la suma proyectada de los retornos "
                    f"logarítmicos al cuadrado de los próximos 5 días.</p>",
                    unsafe_allow_html=True,
                )

            with res_col2:
                fig_velas = go.Figure(data=[go.Candlestick(
                    x=df_plot.index,
                    open=df_plot["Open"],
                    high=df_plot["High"],
                    low=df_plot["Low"],
                    close=df_plot["Close"],
                    name="S&P 500",
                    increasing_line_color="#3B82F6",
                    decreasing_line_color="#94A3B8",
                )])
                estilo_grafica(fig_velas, "Contexto de mercado — últimos 60 días", altura=330, leyenda=False)
                # El rangeslider duplica el eje y roba la mitad del alto útil
                fig_velas.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig_velas, use_container_width=True)

        except Exception as e:
            st.error(f"No fue posible generar la predicción: {e}")

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

        # Fila 2: Pronóstico Semanal vs. Realidad
        st.markdown("<p class='section-title'>Pronóstico Semanal</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-caption'>Serie semana a semana: cada punto es la RV pronosticada al cierre del viernes previo frente a la RV realmente observada en esa semana. La última barra corresponde a la semana en curso.</p>", unsafe_allow_html=True)

        try:
            n_sem = st.radio(
                "Semanas a visualizar",
                [8, 13, 26, 52],
                index=2,
                horizontal=True,
                format_func=lambda n: f"{n} semanas",
            )
            hist_sem = historial_semanal(int(n_sem))
            actual = hist_sem.iloc[-1]

            m1, m2, m3 = st.columns(3)
            m1.metric(f"Pronóstico semana en curso ({hist_sem.index[-1]:%d/%m})",
                      f"{actual['Pronostico']:.4f}")
            if actual["Sesiones"] > 0:
                etiqueta = "RV observada (semana completa)" if actual["Completa"] else f"RV observada ({int(actual['Sesiones'])} de 5 sesiones)"
                m2.metric(etiqueta, f"{actual['Real']:.4f}")
                m3.metric("Desviación vs. pronóstico",
                          f"{(actual['Real'] / actual['Pronostico'] - 1) * 100:+.1f}%",
                          delta_color="off")
            else:
                m2.metric("RV observada", "—")
                m3.metric("Desviación vs. pronóstico", "—")

            completas = hist_sem[hist_sem["Completa"]]
            fig_sem = go.Figure()
            fig_sem.add_trace(go.Scatter(
                x=hist_sem.index, y=hist_sem["Real"], name="RV observada",
                mode="lines+markers",
                line=dict(color="#E2E8F0", width=2.4),
                marker=dict(size=7, color="#E2E8F0"),
                hovertemplate="Semana del %{x|%d/%m/%Y}<br>RV observada: %{y:.4f}<extra></extra>",
            ))
            fig_sem.add_trace(go.Scatter(
                x=hist_sem.index, y=hist_sem["Pronostico"], name="Pronóstico LSTM",
                mode="lines+markers",
                line=dict(color="#3B82F6", width=2.4),
                marker=dict(size=7, color="#3B82F6"),
                hovertemplate="Semana del %{x|%d/%m/%Y}<br>Pronóstico: %{y:.4f}<extra></extra>",
            ))
            estilo_grafica(fig_sem, "Pronóstico vs. volatilidad observada — serie semanal", altura=380)
            fig_sem.update_yaxes(tickformat=".4f", rangemode="tozero")
            fig_sem.update_xaxes(tickformat="%d/%m/%y")
            st.plotly_chart(fig_sem, use_container_width=True)

            if len(completas) >= 2:
                err_rel = (completas["Real"] - completas["Pronostico"]).abs() / completas["Real"]
                err_naive = (completas["Real"] - completas["Persistencia"]).abs() / completas["Real"]
                aciertos = float((err_rel <= 0.20).mean() * 100)
                st.markdown(
                    f"<p class='section-caption'>Sobre las {len(completas)} semanas completas mostradas: "
                    f"error medio de <b>{err_rel.mean() * 100:.1f}%</b> "
                    f"(persistencia: {err_naive.mean() * 100:.1f}%) y "
                    f"<b>{aciertos:.0f}%</b> de las semanas pronosticadas con menos de 20% de desviación. "
                    f"El último punto de la serie observada corresponde a la semana en curso, aún incompleta.</p>",
                    unsafe_allow_html=True,
                )

                # --- Evolución del error porcentual (MAPE) semana a semana ---
                ape = err_rel * 100
                mape_movil = ape.rolling(window=8, min_periods=3).mean()

                fig_mape = go.Figure()
                fig_mape.add_trace(go.Scatter(
                    x=ape.index, y=ape.values, name="Error de la semana (APE)",
                    mode="lines+markers",
                    line=dict(color="#93C5FD", width=1.4),
                    marker=dict(size=5, color="#93C5FD"),
                    hovertemplate="Semana del %{x|%d/%m/%Y}<br>Error: %{y:.1f}%<extra></extra>",
                ))
                fig_mape.add_trace(go.Scatter(
                    x=mape_movil.index, y=mape_movil.values, name="MAPE móvil (8 semanas)",
                    mode="lines",
                    line=dict(color="#3B82F6", width=2.8),
                    hovertemplate="Semana del %{x|%d/%m/%Y}<br>MAPE móvil: %{y:.1f}%<extra></extra>",
                ))
                # "Global" sería engañoso: este promedio solo cubre las semanas
                # visibles y cambia con el selector de arriba.
                fig_mape.add_hline(
                    y=float(ape.mean()), line_dash="dash", line_color="#E2E8F0",
                    annotation_text=f"MAPE de las {len(completas)} semanas mostradas: {ape.mean():.1f}%",
                    annotation_position="top left", annotation_font_color="#E2E8F0",
                )
                fig_mape.add_hrect(y0=0, y1=20, fillcolor="rgba(34,197,94,0.08)", line_width=0)
                estilo_grafica(fig_mape, "Evolución del error de pronóstico (MAPE) por semana", altura=340)
                fig_mape.update_yaxes(ticksuffix="%", rangemode="tozero")
                fig_mape.update_xaxes(tickformat="%d/%m/%y")
                st.plotly_chart(fig_mape, use_container_width=True)

                st.markdown(
                    "<p class='section-caption'>La línea clara es el error de cada semana y la azul su promedio móvil de 8 semanas, que revela si el modelo mejora o se degrada con el tiempo. "
                    "La banda verde marca la zona objetivo (por debajo del 20% de error).<br>"
                    "Este error semanal <b>no es comparable</b> con el MAPE diario de la sección siguiente y suele ser menor: "
                    "aquí hay un pronóstico por semana (sin solapamiento) sobre el periodo visible, mientras que abajo hay uno por día "
                    "sobre dos años, y cada episodio de volatilidad se cuenta unas cinco veces.</p>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.warning(f"No fue posible generar el pronóstico semanal: {e}")

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

        # Fila 3: Desempeño del pronóstico (calculado en vivo, no valores fijos)
        st.markdown("<p class='section-title'>Desempeño del Pronóstico</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-caption'>Predicciones reconstruidas <b>día a día</b> sobre los últimos 2 años (~500 ventanas solapadas) frente a la volatilidad realizada observada. El baseline de comparación es la persistencia: suponer que la volatilidad de la próxima semana será igual a la de la semana pasada.</p>", unsafe_allow_html=True)

        try:
            with st.spinner("Evaluando el modelo sobre datos históricos..."):
                df_eval, metricas = evaluar_desempeno_historico(anios=2)

            mm, mp = metricas["modelo"], metricas["persistencia"]
            col1, col2, col3, col4 = st.columns(4)
            # delta_color="inverse": en métricas de error, menos es mejor
            col1.metric("MAPE diario (2 años)", f"{mm['MAPE']:.1f}%",
                        delta=f"{mm['MAPE'] - mp['MAPE']:+.1f} pp vs. persistencia",
                        delta_color="inverse")
            col2.metric("MdAPE (mediana)", f"{mm['MdAPE']:.1f}%")
            col3.metric("RMSE", f"{mm['RMSE']:.5f}")
            col4.metric("QLIKE", f"{mm['QLIKE']:.3f}",
                        delta=f"{mm['QLIKE'] - mp['QLIKE']:+.3f} vs. persistencia",
                        delta_color="inverse")

            st.markdown("<br>", unsafe_allow_html=True)

            # Gráfica principal: volatilidad real vs predicha
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=df_eval.index, y=df_eval["Real"],
                mode="lines", name="Volatilidad Realizada (Real)",
                line=dict(color="#E2E8F0", width=1.8),
            ))
            fig_perf.add_trace(go.Scatter(
                x=df_eval.index, y=df_eval["Predicción"],
                mode="lines", name="Pronóstico HAR-X + LSTM",
                line=dict(color="#3B82F6", width=1.8),
            ))
            # Baseline visible: sin él, ninguna curva de pronóstico se puede juzgar
            fig_perf.add_trace(go.Scatter(
                x=df_eval.index, y=df_eval["Persistencia"],
                mode="lines", name="Persistencia (baseline)",
                line=dict(color="#94A3B8", width=1.1, dash="dot"),
                opacity=0.75,
            ))
            estilo_grafica(fig_perf, "Volatilidad Real vs. Predicha (RV a 5 días)", altura=400)
            fig_perf.update_yaxes(tickformat=".4f")
            st.plotly_chart(fig_perf, use_container_width=True)

        except Exception as e:
            st.warning(f"No fue posible calcular el desempeño histórico: {e}")


# --- CONTENIDO PESTAÑA 2 ---
with tab2:
    st.markdown("""
        <h2>Panel de Control</h2>
        <p style="color: #A0B2C6; font-size: 1.1rem;">Análisis descriptivo de las variables de mercado que originan las predictoras del modelo LSTM.</p>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Cargando datos del mercado..."):
            df_dash = cargar_datos_dashboard()
    except Exception as e:
        st.warning(f"No fue posible cargar los datos del dashboard: {e}")
    else:
        # ---------- Único filtro: horizonte temporal ----------
        rango = st.radio(
            "Horizonte temporal",
            ["3M", "6M", "1A", "2A", "5A"],
            index=2,
            horizontal=True,
        )
        dias_rango = {"3M": 91, "6M": 182, "1A": 365, "2A": 730, "5A": 1825}
        corte = df_dash.index.max() - pd.Timedelta(days=dias_rango[rango])
        d = df_dash[df_dash.index >= corte]

        # ---------- KPIs del último cierre ----------
        ultimo, previo = d.iloc[-1], d.iloc[-2]
        estado_rsi = "Sobrecompra" if ultimo["RSI"] > 70 else ("Sobreventa" if ultimo["RSI"] < 30 else "Neutral")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("S&P 500 (Close)", f"{ultimo['Close']:,.0f}",
                  delta=f"{(ultimo['Close'] / previo['Close'] - 1) * 100:+.2f}%")
        k2.metric("RSI (14)", f"{ultimo['RSI']:.1f}", delta=estado_rsi, delta_color="off")
        k3.metric("Vol. 20d anualizada", f"{ultimo['Vol_20d'] * np.sqrt(252) * 100:.1f}%")
        k4.metric("ATR (14)", f"{ultimo['ATR']:,.1f} pts")
        k5.metric("Momentum (ROC 10)", f"{ultimo['Momento']:+.2f}%")

        st.markdown(f"<p class='section-caption'>Último dato: {d.index.max().strftime('%d/%m/%Y')} · {len(d):,} sesiones en el rango seleccionado</p>", unsafe_allow_html=True)

        # ---------- Precio + Volumen ----------
        color_vol = np.where(d["Close"].diff() >= 0, "#3B82F6", "#64748B")
        fig_precio = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.72, 0.28], vertical_spacing=0.06)
        fig_precio.add_trace(go.Scatter(
            x=d.index, y=d["High"], name="High", mode="lines",
            line=dict(color="rgba(148,163,184,0)", width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig_precio.add_trace(go.Scatter(
            x=d.index, y=d["Low"], name="Rango High-Low", mode="lines",
            line=dict(color="rgba(148,163,184,0)", width=0),
            fill="tonexty", fillcolor="rgba(59,130,246,0.12)",
        ), row=1, col=1)
        fig_precio.add_trace(go.Scatter(
            x=d.index, y=d["Close"], name="Close",
            line=dict(color="#3B82F6", width=2),
        ), row=1, col=1)
        fig_precio.add_trace(go.Scatter(
            x=d.index, y=d["Open"], name="Open",
            line=dict(color="#E2E8F0", width=1, dash="dot"), visible="legendonly",
        ), row=1, col=1)
        fig_precio.add_trace(go.Bar(
            x=d.index, y=d["Volume"], name="Volumen",
            marker_color=color_vol, marker_line_width=0,
        ), row=2, col=1)
        estilo_grafica(fig_precio, "Precio (OHLC) y Volumen", altura=460)
        st.plotly_chart(fig_precio, use_container_width=True)

        # ---------- Volatilidades + Distribución de retornos ----------
        c1, c2 = st.columns([3, 2])
        with c1:
            fig_vols = go.Figure()
            for col, color, ancho in [("Vol_5d", "#93C5FD", 1.2), ("Vol_20d", "#3B82F6", 2), ("Vol_60d", "#E2E8F0", 1.5)]:
                fig_vols.add_trace(go.Scatter(
                    x=d.index, y=d[col], name=col.replace("Vol_", "Volatilidad "),
                    line=dict(color=color, width=ancho),
                ))
            estilo_grafica(fig_vols, "Volatilidades móviles (5 / 20 / 60 días)")
            st.plotly_chart(fig_vols, use_container_width=True)
        with c2:
            ret_pct = d["Log_Ret"] * 100
            fig_hist = go.Figure(go.Histogram(
                x=ret_pct, nbinsx=60, marker_color="#3B82F6", opacity=0.85, name="Retornos",
            ))
            fig_hist.add_vline(x=float(ret_pct.mean()), line_dash="dash", line_color="#E2E8F0",
                               annotation_text=f"μ = {ret_pct.mean():.3f}%", annotation_font_color="#E2E8F0")
            estilo_grafica(fig_hist, "Distribución de retornos log (%)", leyenda=False)
            fig_hist.update_layout(hovermode="closest")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ---------- Indicadores técnicos: RSI | MACD | Momentum ----------
        t1, t2, t3 = st.columns(3)
        with t1:
            fig_rsi = go.Figure(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
                                           line=dict(color="#3B82F6", width=1.8)))
            fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.10)", line_width=0)
            fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.10)", line_width=0)
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            estilo_grafica(fig_rsi, "RSI (14) — zonas 30/70", altura=300)
            fig_rsi.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_rsi, use_container_width=True)
        with t2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(
                x=d.index, y=d["MACD"] - d["MACD_Senal"], name="Histograma",
                marker_color=np.where(d["MACD"] >= d["MACD_Senal"], "rgba(59,130,246,0.5)", "rgba(148,163,184,0.5)"),
                marker_line_width=0,
            ))
            fig_macd.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
                                          line=dict(color="#3B82F6", width=1.8)))
            fig_macd.add_trace(go.Scatter(x=d.index, y=d["MACD_Senal"], name="Señal",
                                          line=dict(color="#E2E8F0", width=1.2, dash="dot")))
            estilo_grafica(fig_macd, "MACD (12, 26, 9)", altura=300)
            st.plotly_chart(fig_macd, use_container_width=True)
        with t3:
            fig_mom = go.Figure(go.Scatter(
                x=d.index, y=d["Momento"], name="Momentum",
                line=dict(color="#93C5FD", width=1.8), fill="tozeroy",
                fillcolor="rgba(59,130,246,0.10)",
            ))
            fig_mom.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            estilo_grafica(fig_mom, "Momentum — ROC 10 días (%)", altura=300, leyenda=False)
            st.plotly_chart(fig_mom, use_container_width=True)

        # ---------- ATR | Variación del volumen ----------
        a1, a2 = st.columns(2)
        with a1:
            fig_atr = go.Figure(go.Scatter(x=d.index, y=d["ATR"], name="ATR",
                                           line=dict(color="#E2E8F0", width=1.8)))
            estilo_grafica(fig_atr, "ATR (14) — rango verdadero promedio (pts)", altura=300, leyenda=False)
            st.plotly_chart(fig_atr, use_container_width=True)
        with a2:
            fig_vpc = go.Figure(go.Bar(
                x=d.index, y=d["Vol_Pct_Change"] * 100, name="Δ Volumen",
                marker_color=np.where(d["Vol_Pct_Change"] >= 0, "rgba(59,130,246,0.6)", "rgba(148,163,184,0.6)"),
                marker_line_width=0,
            ))
            estilo_grafica(fig_vpc, "Variación diaria del volumen (%)", altura=300, leyenda=False)
            st.plotly_chart(fig_vpc, use_container_width=True)

        # ---------- Matriz de correlación de las variables REALES del modelo ----------
        # Se calculan con el mismo pipeline que usa el pronóstico, no con los
        # indicadores visuales de arriba: la matriz debe describir lo que el
        # modelo efectivamente consume.
        feats_modelo = predictor.preparar(df_dash)
        feats_modelo = feats_modelo[feats_modelo.index >= corte]
        cols_predictoras = predictor.columnas_features
        corr = feats_modelo[cols_predictoras].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=cols_predictoras, y=cols_predictoras,
            zmin=-1, zmax=1,
            colorscale=[[0.0, "#1E293B"], [0.5, "#F4F6F9"], [1.0, "#3B82F6"]],
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(tickfont=dict(color="white"), outlinewidth=0),
        ))
        estilo_grafica(fig_corr, f"Matriz de correlación de las {len(cols_predictoras)} variables del modelo", altura=560, leyenda=False)
        fig_corr.update_layout(hovermode="closest")
        fig_corr.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- CONTENIDO PESTAÑA 3: SELECCIÓN DE MODELOS ---
with tab3:
    st.markdown("""
        <h2>Selección de Modelos</h2>
        <p style="color: #A0B2C6; font-size: 1.1rem;">Comparativa out-of-sample de la red LSTM frente a los modelos econométricos de referencia: HAR-RV, GARCH(1,1) y EGARCH. Todos pronostican el mismo objetivo — la volatilidad realizada de los próximos 5 días hábiles — con parámetros estimados únicamente con información anterior al periodo evaluado.</p>
    """, unsafe_allow_html=True)

    if not modelo_cargado:
        st.error(f"No se pudo cargar el modelo LSTM, la comparativa no está disponible. Detalle: {error_msg}")
    else:
        anios_eval = st.radio(
            "Periodo de evaluación",
            [1, 2, 3],
            index=1,
            horizontal=True,
            format_func=lambda n: f"Último año" if n == 1 else f"Últimos {n} años",
        )

        try:
            with st.spinner("Estimando HAR-RV, GARCH(1,1) y EGARCH, y reconstruyendo las predicciones de la LSTM..."):
                df_comp, met_comp = comparativa_modelos(int(anios_eval))
        except Exception as e:
            st.warning(f"No fue posible construir la comparativa de modelos: {e}")
        else:
            modelos = list(COLORES_MODELOS)
            ranking = sorted(modelos, key=lambda m: met_comp[m]["MAPE"])
            mejor = ranking[0]

            # ---------- Tarjetas: MAPE por modelo ----------
            st.markdown("<p class='section-title'>Error porcentual medio (MAPE)</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p class='section-caption'>Sobre {len(df_comp):,} pronósticos diarios "
                f"{'del último año' if anios_eval == 1 else f'de los últimos {anios_eval} años'} "
                f"({df_comp.index.min():%d/%m/%Y} — {df_comp.index.max():%d/%m/%Y}). "
                f"Mejor modelo del periodo: <b>{mejor}</b>.</p>",
                unsafe_allow_html=True,
            )
            cols_mape = st.columns(4)
            for col, m in zip(cols_mape, modelos):
                if m == mejor:
                    # El distintivo del ganador va en la banda del delta, con el
                    # mismo verde que el resto de la app usa para "esto es bueno".
                    # `delta_color="normal"` lo pinta en verde porque el texto no
                    # empieza con signo negativo; la flecha se oculta por CSS,
                    # que aquí no aportaría nada.
                    with col.container(key="metrica_mejor"):
                        st.metric(m, f"{met_comp[m]['MAPE']:.1f}%",
                                  delta="Mejor desempeño", delta_color="normal")
                else:
                    dif = met_comp[m]["MAPE"] - met_comp[mejor]["MAPE"]
                    col.metric(m, f"{met_comp[m]['MAPE']:.1f}%",
                               delta=f"{dif:+.1f} pp vs. {mejor}",
                               delta_color="inverse")

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------- Tabla completa de métricas ----------
            st.markdown("<p class='section-title'>Resumen de métricas</p>", unsafe_allow_html=True)
            tabla = pd.DataFrame(met_comp).T[["MAPE", "MdAPE", "RMSE", "MAE", "QLIKE"]]
            tabla.index.name = "Modelo"
            st.dataframe(
                tabla.style.format({
                    "MAPE": "{:.2f}%", "MdAPE": "{:.2f}%",
                    "RMSE": "{:.5f}", "MAE": "{:.5f}", "QLIKE": "{:.4f}",
                }).highlight_min(axis=0, props="color: #3B82F6; font-weight: bold;"),
                use_container_width=True,
            )
            st.markdown(
                "<p class='section-caption'>En todas las métricas, menor es mejor, y el mejor valor de cada "
                "columna aparece resaltado. MAPE y MdAPE miden el error relativo (media y mediana), RMSE y MAE "
                "el error absoluto en niveles de RV, y QLIKE es la pérdida estándar de la literatura de "
                "volatilidad. Las definiciones formales están más abajo.</p>",
                unsafe_allow_html=True,
            )

            st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

            # ---------- Serie temporal: real vs. cada modelo ----------
            st.markdown("<p class='section-title'>Pronósticos frente a la volatilidad observada</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-caption'>Cada curva es el pronóstico diario del modelo para la RV de los 5 días siguientes. Haz clic en la leyenda para aislar modelos.</p>", unsafe_allow_html=True)

            fig_series = go.Figure()
            fig_series.add_trace(go.Scatter(
                x=df_comp.index, y=df_comp["Real"],
                mode="lines", name="Volatilidad Realizada (Real)",
                line=dict(color="#E2E8F0", width=2.2),
            ))
            for m in modelos:
                fig_series.add_trace(go.Scatter(
                    x=df_comp.index, y=df_comp[m],
                    mode="lines", name=m,
                    line=dict(color=COLORES_MODELOS[m], width=1.5),
                    opacity=0.9,
                ))
            estilo_grafica(fig_series, "RV a 5 días: observada vs. pronosticada por cada modelo", altura=420)
            fig_series.update_yaxes(tickformat=".4f")
            st.plotly_chart(fig_series, use_container_width=True)

            # ---------- MAPE móvil por modelo ----------
            # Va junto a la serie anterior: ambas son lecturas temporales, y
            # separarlas con los gráficos de barras rompería esa continuidad.
            st.markdown("<p class='section-title'>Estabilidad del error en el tiempo</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-caption'>MAPE móvil de 63 sesiones (~un trimestre): revela si la ventaja de un modelo es estable o depende del régimen de volatilidad.</p>", unsafe_allow_html=True)

            fig_movil = go.Figure()
            for m in modelos:
                ape_m = (df_comp["Real"] - df_comp[m]).abs() / df_comp["Real"] * 100
                fig_movil.add_trace(go.Scatter(
                    x=df_comp.index, y=ape_m.rolling(63, min_periods=21).mean(),
                    mode="lines", name=m,
                    line=dict(color=COLORES_MODELOS[m], width=2),
                ))
            estilo_grafica(fig_movil, "MAPE móvil (63 sesiones) por modelo", altura=380)
            fig_movil.update_yaxes(ticksuffix="%", rangemode="tozero")
            st.plotly_chart(fig_movil, use_container_width=True)

            st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

            # ---------- Gráficos de barras: métrica por métrica ----------
            st.markdown("<p class='section-title'>Comparativa métrica por métrica</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-caption'>La misma información del resumen, en forma visual: cada barra es un modelo y en todos los paneles la barra más baja es la mejor.</p>", unsafe_allow_html=True)

            g1, g2 = st.columns([3, 2])
            with g1:
                fig_bar = go.Figure()
                for metrica, opac in [("MAPE", 1.0), ("MdAPE", 0.55)]:
                    fig_bar.add_trace(go.Bar(
                        x=modelos,
                        y=[met_comp[m][metrica] for m in modelos],
                        name=f"{metrica} (%)",
                        marker_color=[COLORES_MODELOS[m] for m in modelos],
                        marker_line_width=0,
                        opacity=opac,
                        text=[f"{met_comp[m][metrica]:.1f}%" for m in modelos],
                        textposition="outside",
                        textfont=dict(color="white"),
                    ))
                estilo_grafica(fig_bar, "Error porcentual: media (MAPE) y mediana (MdAPE)", altura=380)
                fig_bar.update_layout(barmode="group")
                fig_bar.update_yaxes(ticksuffix="%", rangemode="tozero")
                st.plotly_chart(fig_bar, use_container_width=True)
            with g2:
                # QLIKE: la pérdida canónica de la literatura de volatilidad
                fig_ql = go.Figure(go.Bar(
                    x=modelos,
                    y=[met_comp[m]["QLIKE"] for m in modelos],
                    marker_color=[COLORES_MODELOS[m] for m in modelos],
                    marker_line_width=0,
                    text=[f"{met_comp[m]['QLIKE']:.3f}" for m in modelos],
                    textposition="outside",
                    textfont=dict(color="white"),
                ))
                estilo_grafica(fig_ql, "QLIKE (pérdida robusta, menor = mejor)", altura=380, leyenda=False)
                st.plotly_chart(fig_ql, use_container_width=True)

            # ---------- Barras: RMSE y MAE ----------
            fig_err = make_subplots(rows=1, cols=2, subplot_titles=("RMSE", "MAE"),
                                    horizontal_spacing=0.08)
            for j, metrica in enumerate(["RMSE", "MAE"], start=1):
                fig_err.add_trace(go.Bar(
                    x=modelos,
                    y=[met_comp[m][metrica] for m in modelos],
                    marker_color=[COLORES_MODELOS[m] for m in modelos],
                    marker_line_width=0,
                    text=[f"{met_comp[m][metrica]:.5f}" for m in modelos],
                    textposition="outside",
                    textfont=dict(color="white", size=10),
                    showlegend=False,
                ), row=1, col=j)
            estilo_grafica(fig_err, "Errores absolutos sobre niveles de RV (menor = mejor)", altura=360, leyenda=False)
            # Los subtítulos de cada panel viven en el margen superior; sin
            # bajarlos quedan pegados al título principal.
            fig_err.update_annotations(font_color="#E2E8F0", font_size=13, yshift=-14)
            fig_err.update_yaxes(tickformat=".4f", rangemode="tozero")
            st.plotly_chart(fig_err, use_container_width=True)

            st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

            # ---------- Definición formal de las métricas ----------
            st.markdown("<p class='section-title'>Cómo se calcula cada métrica</p>", unsafe_allow_html=True)
            st.markdown(
                "<p class='section-caption'>Donde <b>n</b> es el número de pronósticos del periodo, "
                "<b>RV<sub>t</sub></b> la volatilidad realizada observada en la fecha t y "
                "<b>RV̂<sub>t</sub></b> la pronosticada por el modelo. En las cinco, un valor menor indica "
                "un mejor pronóstico.</p>",
                unsafe_allow_html=True,
            )

            anchos = [1.15, 2.5, 3.0]
            with st.container(border=True, key="tabla_formulas"):
                enc = st.columns(anchos, gap="medium")
                for col, titulo_col in zip(enc, ["Métrica", "Definición", "Qué mide y cuándo importa"]):
                    col.markdown(f"<p class='tabla-encabezado'>{titulo_col}</p>", unsafe_allow_html=True)

                for i, (sigla, nombre, formula, lectura) in enumerate(FORMULAS_METRICAS):
                    st.markdown("<hr class='fila-sep'>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(anchos, gap="medium")
                    c1.markdown(
                        f"<p class='formula-sigla'>{sigla}</p>"
                        f"<p class='formula-nombre'>{nombre}</p>",
                        unsafe_allow_html=True,
                    )
                    with c2:
                        st.latex(formula)
                    c3.markdown(f"<p class='formula-lectura'>{lectura}</p>", unsafe_allow_html=True)

            st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

            # ---------- Fichas de los modelos comparados ----------
            st.markdown("<p class='section-title'>Los contendientes</p>", unsafe_allow_html=True)
            fm1, fm2, fm3, fm4 = st.columns(4, gap="medium")
            with fm1:
                st.markdown("""
                    <div class="info-card">
                        <h4>LSTM (HAR-X + residual)</h4>
                        <p>Red recurrente que corrige el residual de un HAR-X con VIX. Ventanas de 22 días
                        y 13 variables estacionarias. Es el modelo en producción de Volatitlán.</p>
                    </div>
                """, unsafe_allow_html=True)
            with fm2:
                st.markdown("""
                    <div class="info-card">
                        <h4>HAR-RV — Corsi (2009)</h4>
                        <p>Regresión lineal de la RV futura sobre sus componentes diario, semanal y mensual.
                        Simple, robusto y notoriamente difícil de batir en horizontes cortos.</p>
                    </div>
                """, unsafe_allow_html=True)
            with fm3:
                st.markdown("""
                    <div class="info-card">
                        <h4>GARCH(1,1) — Bollerslev (1986)</h4>
                        <p>Varianza condicional con persistencia simétrica: el estándar econométrico
                        durante décadas. La varianza semanal se agrega desde los pronósticos diarios.</p>
                    </div>
                """, unsafe_allow_html=True)
            with fm4:
                st.markdown("""
                    <div class="info-card">
                        <h4>EGARCH(1,1) — Nelson (1991)</h4>
                        <p>GARCH en logaritmos con término asimétrico: captura el efecto apalancamiento.
                        Su pronóstico multi-paso se obtiene por simulación (1.000 trayectorias).</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(
                "<br><p class='section-caption'><b>Protocolo:</b> los parámetros de HAR-RV, GARCH y EGARCH se estiman "
                "con todo el histórico disponible desde 1990 hasta el inicio del periodo de evaluación y quedan congelados; "
                "después solo se filtran con la información de cada día. La LSTM también fue entrenada con datos anteriores "
                "al periodo evaluado. Ningún modelo ve el futuro que se le pide pronosticar.</p>",
                unsafe_allow_html=True,
            )

# --- CONTENIDO PESTAÑA 4 ---
with tab4:
    st.markdown("<h2>Acerca del Autor</h2>", unsafe_allow_html=True)

    perfil_col, texto_col = st.columns([1, 2.4], gap="large")

    with perfil_col:
        ruta_foto = _CARPETA_APP / "cris_ML.jpeg"
        if ruta_foto.exists():
            st.markdown('<div class="autor-foto">', unsafe_allow_html=True)
            st.image(str(ruta_foto), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='info-card' style='text-align:center;'><p>Fotografía no encontrada<br><code>cris_ML.jpeg</code></p></div>",
                unsafe_allow_html=True,
            )

    with texto_col:
        st.markdown("""
            <div class="autor-nombre">Cristian A. Materanda</div>
            <div class="autor-rol">Matemático · Científico de Datos · Lógico Estadístico</div>
            <p class="autor-bio">
                Mi labor profesional es presentar información estratégica obtenida a partir del
                análisis de datos, la modelación estadística y el machine learning, con el objetivo
                de tomar decisiones concluyentemente exitosas en objetivos de negocio.
            </p>
            <div class="contacto-grid">
                <a class="contacto-chip" href="https://www.linkedin.com/in/mlmate" target="_blank">LinkedIn</a>
                <a class="contacto-chip" href="https://www.linkedin.com/in/crismate" target="_blank">LinkedIn ML</a>
                <a class="contacto-chip" href="https://drive.google.com/file/d/1ZUjyahRFct0BQFL-7nzfC3NKsLrb7aU9/view?usp=drive_link" target="_blank">Curriculum Vitae</a>
                <a class="contacto-chip" href="mailto:cm180140@gmail.com">cm180140@gmail.com</a>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

    # ---------- Ficha técnica del proyecto ----------
    st.markdown("<p class='section-title'>Sobre este proyecto</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-caption'>Volatitlán es una plataforma de pronóstico de volatilidad financiera construida sobre redes neuronales recurrentes.</p>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        st.markdown("""
            <div class="info-card">
                <h4>Objetivo</h4>
                <p>Predecir la volatilidad realizada a cinco días del índice S&P 500 mediante una
                arquitectura LSTM rigurosamente regularizada, superando la capacidad predictiva
                de los modelos econométricos tradicionales.</p>
            </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
            <div class="info-card">
                <h4>Metodología</h4>
                <ul>
                    <li>Ventanas deslizantes de 60 días</li>
                    <li>11 variables predictoras estacionarias</li>
                    <li>LSTM (64) + Dropout + Densa (32)</li>
                    <li>Validación temporal sin data leakage</li>
                    <li>Early Stopping sobre val_loss</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
            <div class="info-card">
                <h4>Stack Tecnológico</h4>
                <ul>
                    <li>TensorFlow / Keras</li>
                    <li>scikit-learn · pandas · NumPy</li>
                    <li>yfinance · ta</li>
                    <li>Plotly · Streamlit</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2, gap="medium")
    with r1:
        st.markdown("""
            <div class="info-card">
                <h4>Referencias</h4>
                <ul>
                    <li>Hochreiter & Schmidhuber (1997). <i>Long Short-Term Memory</i>.</li>
                    <li>Bollerslev (1986). <i>Generalized Autoregressive Conditional Heteroskedasticity</i>.</li>
                    <li>Nelson (1991). <i>Conditional Heteroskedasticity in Asset Returns</i>.</li>
                    <li>Corsi (2009). <i>A Simple Approximate Long-Memory Model of Realized Volatility</i>.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
            <div class="info-card">
                <h4>Aviso</h4>
                <p>Este proyecto tiene fines académicos y demostrativos. Las predicciones aquí
                presentadas no constituyen asesoría financiera ni recomendación de inversión.
                Los datos provienen de Yahoo Finance y el desempeño histórico no garantiza
                resultados futuros.</p>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# 6. Footer
# ==========================================
st.markdown('<div class="footer-signature">Martingale Lab</div>', unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from pathlib import Path

# Carpeta donde vive app.py (para localizar recursos como la fotografía)
_CARPETA_APP = Path(__file__).resolve().parent
from datetime import datetime, timedelta
# Pipeline de inferencia y descarga de datos (mismos módulos que el entrenamiento)
from Prediccion import PredictorVolatilidad
from Caracteristicas import descargar_datos

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

    .footer-signature { font-family: 'Great Vibes', cursive !important; font-size: 2.2rem; text-align: right; color: #64748B; margin-top: 2rem; padding-right: 2rem; }
    </style>
""", unsafe_allow_html=True)

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
    hoy = datetime.today()
    # Buffer amplio: la RV a 66 días y las medias móviles de 22 consumen ~90
    # sesiones de calentamiento antes de la primera fila utilizable.
    inicio = hoy - timedelta(days=int(anios * 365) + 320)

    df = descargar_datos(inicio, hoy + timedelta(days=1))
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
    hoy = datetime.today()
    inicio = hoy - timedelta(days=6 * 365)
    df = descargar_datos(inicio, hoy + timedelta(days=1))

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


def estilo_grafica(fig, titulo, altura=320):
    """Aplica el tema azul marino consistente a cualquier figura Plotly."""
    fig.update_layout(
        title=titulo,
        height=altura,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        font=dict(color="white", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


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
tab1, tab2, tab3 = st.tabs(["Pronóstico", "Dashboard", "Acerca del Autor"])

# --- CONTENIDO PESTAÑA 1: PRONÓSTICO Y MÉTRICAS ---
with tab1:
    st.markdown("""
        <h2>Pronóstico LSTM</h2>
        <p style="color: #A0B2C6; font-size: 1.1rem;">Análisis y predicción de la volatilidad del índice S&P 500 utilizando redes neuronales recurrentes.</p>
    """, unsafe_allow_html=True)
    
    if not modelo_cargado:
        st.error(f"⚠️ No se pudo cargar el modelo. Verifica que 'modelo_lstm_sp500_volatilidad.keras' y los escaladores existan. Detalle: {error_msg}")
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
                    f"⚠️ Yahoo Finance no respondió; se está usando la copia local del "
                    f"histórico, con datos hasta el {_ultimo:%d/%m/%Y} ({_dias} días). "
                    f"Los pronósticos no incorporan las sesiones más recientes."
                )
            elif _dias > 5:
                st.info(
                    f"ℹ️ Último dato de mercado disponible: {_ultimo:%d/%m/%Y} "
                    f"({_dias} días de antigüedad)."
                )
        except Exception:
            pass  # el aviso es accesorio: nunca debe tumbar la página

        # Fila 1: Desempeño del pronóstico (calculado en vivo, no valores fijos)
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
            fig_perf.update_layout(
                title="Volatilidad Real vs. Predicha (RV a 5 días)",
                margin=dict(l=20, r=20, t=50, b=20),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.1)",
                font=dict(color="white"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickformat=".4f"),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        except Exception as e:
            st.warning(f"No fue posible calcular el desempeño histórico: {e}")

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

        # Fila 1.5: Pronóstico Semanal vs. Realidad
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
                    f"<p class='section-caption'>📊 Sobre las {len(completas)} semanas completas mostradas: "
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
                    "<p class='section-caption'>💡 La línea clara es el error de cada semana y la azul su promedio móvil de 8 semanas, que revela si el modelo mejora o se degrada con el tiempo. "
                    "La banda verde marca la zona objetivo (por debajo del 20% de error).<br>"
                    "⚠️ Este error semanal <b>no es comparable</b> con el MAPE diario de la sección anterior y suele ser menor: "
                    "aquí hay un pronóstico por semana (sin solapamiento) sobre el periodo visible, mientras que arriba hay uno por día "
                    "sobre dos años, y cada episodio de volatilidad se cuenta unas cinco veces.</p>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.warning(f"No fue posible generar el pronóstico semanal: {e}")

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

        # Fila 2: Botón de Predicción y Resultados
        st.markdown("<p class='section-title'>Consultar Pronóstico Actualizado</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-caption'>Descarga los datos más recientes del mercado y genera la predicción de volatilidad para los próximos 5 días hábiles.</p>", unsafe_allow_html=True)

        if st.button("🚀 Generar Predicción a 5 Días", use_container_width=True):
            with st.spinner(f'Conectando con Yahoo Finance y procesando tensores de {predictor.ventana} días...'):
                try:
                    # Ejecutar inferencia
                    volatilidad_esperada = predictor.predecir_futuro()
                    df_reciente = predictor.obtener_datos_recientes()
                    df_plot = df_reciente.tail(60)  # contexto visual, más amplio que la ventana

                    # Mostrar el resultado destacado
                    st.success("✅ Inferencia completada con éxito.")

                    res_col1, res_col2 = st.columns([1, 2])

                    with res_col1:
                        st.metric(
                            label="Volatilidad Realizada (Próximos 5 días)",
                            value=f"{volatilidad_esperada:.4f}"
                        )
                        st.info("💡 Este valor es la raíz cuadrada de la suma proyectada de los retornos logarítmicos al cuadrado de los próximos 5 días (volatilidad realizada).")

                    with res_col2:
                        # Gráfica de contexto con Plotly (Estilo oscuro para combinar)
                        fig = go.Figure(data=[go.Candlestick(
                            x=df_plot.index,
                            open=df_plot['Open'],
                            high=df_plot['High'],
                            low=df_plot['Low'],
                            close=df_plot['Close'],
                            name="S&P 500",
                            increasing_line_color="#3B82F6",
                            decreasing_line_color="#94A3B8",
                        )])
                        
                        fig.update_layout(
                            title="Ventana de Entrada del Modelo (Últimos 60 días)",
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=300,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0.1)',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error al generar la predicción: {e}")
                    
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
            estilo_grafica(fig_hist, "Distribución de retornos log (%)")
            fig_hist.update_layout(hovermode="closest", showlegend=False)
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
            estilo_grafica(fig_mom, "Momentum — ROC 10 días (%)", altura=300)
            fig_mom.update_layout(showlegend=False)
            st.plotly_chart(fig_mom, use_container_width=True)

        # ---------- ATR | Variación del volumen ----------
        a1, a2 = st.columns(2)
        with a1:
            fig_atr = go.Figure(go.Scatter(x=d.index, y=d["ATR"], name="ATR",
                                           line=dict(color="#E2E8F0", width=1.8)))
            estilo_grafica(fig_atr, "ATR (14) — rango verdadero promedio (pts)", altura=300)
            fig_atr.update_layout(showlegend=False)
            st.plotly_chart(fig_atr, use_container_width=True)
        with a2:
            fig_vpc = go.Figure(go.Bar(
                x=d.index, y=d["Vol_Pct_Change"] * 100, name="Δ Volumen",
                marker_color=np.where(d["Vol_Pct_Change"] >= 0, "rgba(59,130,246,0.6)", "rgba(148,163,184,0.6)"),
                marker_line_width=0,
            ))
            estilo_grafica(fig_vpc, "Variación diaria del volumen (%)", altura=300)
            fig_vpc.update_layout(showlegend=False)
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
        estilo_grafica(fig_corr, f"Matriz de correlación de las {len(cols_predictoras)} variables del modelo", altura=560)
        fig_corr.update_layout(hovermode="closest")
        fig_corr.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- CONTENIDO PESTAÑA 3 ---
with tab3:
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
                <a class="contacto-chip" href="https://www.linkedin.com/in/mlmate" target="_blank">🔗 LinkedIn</a>
                <a class="contacto-chip" href="https://www.linkedin.com/in/crismate" target="_blank">🤖 LinkedIn ML</a>
                <a class="contacto-chip" href="https://drive.google.com/file/d/1ZUjyahRFct0BQFL-7nzfC3NKsLrb7aU9/view?usp=drive_link" target="_blank">📄 Curriculum Vitae</a>
                <a class="contacto-chip" href="mailto:cm180140@gmail.com">✉️ cm180140@gmail.com</a>
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
                <h4>🎯 Objetivo</h4>
                <p>Predecir la volatilidad realizada a cinco días del índice S&P 500 mediante una
                arquitectura LSTM rigurosamente regularizada, superando la capacidad predictiva
                de los modelos econométricos tradicionales.</p>
            </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
            <div class="info-card">
                <h4>🧠 Metodología</h4>
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
                <h4>🛠️ Stack Tecnológico</h4>
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
                <h4>📚 Referencias</h4>
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
                <h4>⚠️ Aviso</h4>
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
"""
Descarga de datos e ingeniería de características.

FUENTE ÚNICA DE VERDAD: tanto `Entrenamiento.py` como `Prediccion.py` importan
de aquí. Antes cada uno tenía su propia copia del cálculo de features, lo que
permite que entrenamiento y producción diverjan sin que nada falle de forma
visible (train/serve skew). Con un solo módulo, eso es imposible por
construcción.

Criterios de diseño de las variables:

1. Todas son estacionarias. Los precios y el volumen en niveles rompen el
   escalado cuando el índice sale del rango visto en entrenamiento.
2. Se usa el VIX. Es la única variable exógena y forward-looking disponible:
   el mercado publica todos los días su propio pronóstico de la volatilidad a
   30 días. Ignorarlo y pronosticar el S&P 500 solo con el pasado del S&P 500
   deja fuera la información más relevante del problema.
3. Se prefieren estimadores de rango (Garman-Klass, Parkinson) sobre retornos
   cierre-a-cierre: usan High/Low y son varias veces menos ruidosos con la
   misma cantidad de datos.
4. La asimetría se codifica de forma explícita mediante semivarianzas. Lo que
   predice la volatilidad futura es el efecto apalancamiento (los retornos
   negativos la elevan), no la dirección que miden RSI/MACD.
"""

import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

_CARPETA = Path(__file__).resolve().parent

# Copia local del histórico, versionada en el repositorio. Yahoo Finance
# limita las peticiones que llegan desde IPs de centros de datos, así que en
# un despliegue en la nube la descarga en vivo puede fallar de forma
# permanente. Con este respaldo la app sigue funcionando (con datos hasta la
# fecha del archivo) en lugar de quedar inutilizable.
ARCHIVO_RESPALDO = _CARPETA / "datos_respaldo.csv"

REINTENTOS = 3
ESPERA_REINTENTO = 1.5  # segundos, se duplica en cada intento


# ==========================================
# Configuración del problema
# ==========================================
VENTANA = 22        # pasos temporales que ve la red (un mes bursátil)
HORIZONTE = 5       # días hábiles del target: RV de la próxima semana
FECHA_INICIO = "1990-01-01"  # límite del ^VIX; triplica la muestra de 2015

# Variables que entran a la red
COLUMNAS_X = [
    "Ret_Std",
    "log_RV_1d",
    "log_RV_5d",
    "log_RV_22d",
    "log_RV_66d",
    "log_Semivar_Neg",
    "Asimetria",
    "log_Park_5d",
    "Rango_Rel",
    "Vol_Rel",
    "log_VIX",
    "VIX_Chg",
    "IV_RV_Spread",
]

# Regresores del HAR-X (componente lineal que la red solo corrige)
COLUMNAS_HAR = ["log_RV_1d", "log_RV_5d", "log_RV_22d", "log_VIX"]

# Columnas auxiliares necesarias para benchmarks y para el target
COLUMNAS_AUX = ["Log_Ret", "RV_5d", "RV_22d"]

_EPS = 1e-8


# ==========================================
# 1. Descarga
# ==========================================
def _bajar(ticker, str_inicio, str_fin):
    """Descarga un ticker con reintentos ante límites de tasa transitorios."""
    ultimo_error = None
    for intento in range(REINTENTOS):
        try:
            d = yf.download(ticker, start=str_inicio, end=str_fin,
                            auto_adjust=False, progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                return d
            ultimo_error = ValueError(f"yfinance devolvió un frame vacío para {ticker}.")
        except Exception as e:  # red, rate limit, cambio de esquema...
            ultimo_error = e
        if intento < REINTENTOS - 1:
            time.sleep(ESPERA_REINTENTO * (2 ** intento))
    raise ValueError(f"yfinance no devolvió datos para {ticker}: {ultimo_error}")


def _descargar_yahoo(str_inicio, str_fin):
    """OHLCV del ^GSPC más el cierre del ^VIX, alineados en un solo frame."""
    sp = _bajar("^GSPC", str_inicio, str_fin)
    vix = _bajar("^VIX", str_inicio, str_fin)

    faltantes = {"Open", "High", "Low", "Close", "Volume"}.difference(sp.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en ^GSPC: {sorted(faltantes)}")

    df = sp[["Open", "High", "Low", "Close", "Volume"]].copy()
    # ffill corto: cubre feriados desalineados entre ambos índices sin
    # inventar datos en huecos largos.
    df["VIX"] = vix["Close"].reindex(df.index).ffill(limit=5)
    return df.dropna(subset=["VIX"])


def _cargar_respaldo():
    """Lee la copia local del histórico. Devuelve None si no existe."""
    if not ARCHIVO_RESPALDO.exists():
        return None
    d = pd.read_csv(ARCHIVO_RESPALDO, index_col=0, parse_dates=True)
    return d if not d.empty else None


def descargar_datos(fecha_inicio=FECHA_INICIO, fecha_fin=None, usar_respaldo=True):
    """
    Devuelve OHLCV del ^GSPC + VIX para el rango pedido.

    Combina la descarga en vivo con el respaldo versionado: si Yahoo responde,
    sus filas tienen prioridad y el respaldo solo rellena hacia atrás; si Yahoo
    falla por completo (bloqueo desde IP de nube), la app sigue operando con el
    respaldo en lugar de caerse.

    El frame lleva `df.attrs["fuente"]` con el origen efectivo de los datos
    ("yahoo", "respaldo" o "mixto") para que la interfaz pueda avisarlo.
    """
    if fecha_fin is None:
        fecha_fin = datetime.today() + timedelta(days=1)  # `end` es exclusivo
    ts_inicio, ts_fin = pd.Timestamp(fecha_inicio), pd.Timestamp(fecha_fin)
    str_inicio, str_fin = ts_inicio.strftime("%Y-%m-%d"), ts_fin.strftime("%Y-%m-%d")

    try:
        vivo = _descargar_yahoo(str_inicio, str_fin)
    except Exception as error_vivo:
        vivo = None
        if not usar_respaldo:
            raise
        motivo = error_vivo

    respaldo = _cargar_respaldo() if usar_respaldo else None

    if vivo is None and respaldo is None:
        raise ValueError(
            f"Sin datos: la descarga falló ({motivo}) y no hay respaldo local en "
            f"{ARCHIVO_RESPALDO.name}."
        )

    if vivo is None:
        df, fuente = respaldo.copy(), "respaldo"
    elif respaldo is None:
        df, fuente = vivo, "yahoo"
    else:
        previas = respaldo[~respaldo.index.isin(vivo.index)]
        df = pd.concat([previas, vivo]).sort_index()
        fuente = "mixto" if len(previas) else "yahoo"

    # Autorrefresco del respaldo: cada descarga en vivo exitosa deja el
    # archivo local al día, así el fallback nunca se queda semanas atrás.
    # Es seguro por construcción: el frame fusionado (aún sin recortar al
    # rango pedido) siempre es un superconjunto del respaldo existente.
    if vivo is not None and usar_respaldo:
        try:
            if respaldo is None or df.index.max() > respaldo.index.max():
                df.to_csv(ARCHIVO_RESPALDO)
        except OSError:
            pass  # disco de solo lectura (p. ej. algunos despliegues): no es fatal

    df = df.loc[(df.index >= ts_inicio) & (df.index < ts_fin)]
    if df.empty:
        raise ValueError(f"No hay datos disponibles entre {str_inicio} y {str_fin}.")

    df.attrs["fuente"] = fuente
    return df


def guardar_respaldo(fecha_inicio="2015-01-01", fecha_fin=None):
    """
    Regenera el archivo de respaldo desde Yahoo.

    Ejecutar desde una máquina con acceso (no desde el servidor de despliegue)
    y versionar el resultado. Conviene actualizarlo cada vez que se reentrena.
    """
    df = _descargar_yahoo(
        pd.Timestamp(fecha_inicio).strftime("%Y-%m-%d"),
        pd.Timestamp(fecha_fin or datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    df.to_csv(ARCHIVO_RESPALDO)
    return df


# ==========================================
# 2. Ingeniería de características
# ==========================================
def ingenieria_caracteristicas(df, con_target=True):
    """
    Calcula todas las variables del modelo.

    `con_target=False` para inferencia: el target de la última semana todavía
    no se observa, y exigirlo eliminaría precisamente las filas que se quieren
    predecir.
    """
    d = df.copy()

    # Guarda ante los primeros años de Yahoo, donde el Open del índice puede
    # venir en cero: sin esto, Garman-Klass produce infinitos.
    apertura = d["Open"].where(d["Open"] > 0, d["Close"])

    d["Log_Ret"] = np.log(d["Close"] / d["Close"].shift(1))
    r = d["Log_Ret"]
    r2 = r ** 2

    # --- Estimadores de volatilidad basados en rango -------------------
    log_hl = np.log(d["High"] / d["Low"])
    log_co = np.log(d["Close"] / apertura)
    gk_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    d["log_RV_1d"] = np.log(np.sqrt(gk_var.clip(lower=1e-10)) + _EPS)

    park_var = (log_hl ** 2).rolling(5).mean() / (4 * np.log(2))
    d["log_Park_5d"] = np.log(np.sqrt(park_var.clip(lower=1e-10)) + _EPS)

    # --- Volatilidad realizada acumulada (estructura tipo HAR) ---------
    d["RV_5d"] = np.sqrt(r2.rolling(5).sum())
    d["RV_22d"] = np.sqrt(r2.rolling(22).sum())
    d["RV_66d"] = np.sqrt(r2.rolling(66).sum())
    d["log_RV_5d"] = np.log(d["RV_5d"] + _EPS)
    d["log_RV_22d"] = np.log(d["RV_22d"] + _EPS)
    d["log_RV_66d"] = np.log(d["RV_66d"] + _EPS)

    # --- Asimetría / efecto apalancamiento ------------------------------
    rs_neg = r2.where(r < 0, 0.0).rolling(22).sum()
    rs_pos = r2.where(r > 0, 0.0).rolling(22).sum()
    d["log_Semivar_Neg"] = np.log(np.sqrt(rs_neg.clip(lower=1e-10)) + _EPS)
    d["Asimetria"] = (rs_neg - rs_pos) / (rs_neg + rs_pos + 1e-12)

    # --- Sorpresa del día y actividad -----------------------------------
    # Se estandariza con la volatilidad hasta t-1 para que la variable mida
    # cuán inesperado fue el retorno, no su magnitud absoluta.
    vol_diaria_previa = d["RV_5d"].shift(1) / np.sqrt(5)
    d["Ret_Std"] = (r / (vol_diaria_previa + _EPS)).clip(-10, 10)
    d["Rango_Rel"] = (d["High"] - d["Low"]) / d["Close"]
    d["Vol_Rel"] = d["Volume"] / d["Volume"].rolling(22).mean()

    # --- Volatilidad implícita ------------------------------------------
    d["log_VIX"] = np.log(d["VIX"] / 100.0)
    d["VIX_Chg"] = d["log_VIX"].diff()
    # Prima de riesgo de varianza: implícita (anualizada) menos realizada
    # anualizada. Es el sesgo conocido del VIX sobre la RV futura.
    rv22_anual = d["RV_22d"] * np.sqrt(252.0 / 22.0)
    d["IV_RV_Spread"] = d["log_VIX"] - np.log(rv22_anual + _EPS)

    columnas = COLUMNAS_X + COLUMNAS_AUX + ["RV_66d"]

    if con_target:
        # RV(t+1..t+5): lo que se quiere pronosticar. En log, porque la RV
        # tiene cola pesada y en logs el error optimizado es ~relativo.
        d["Target"] = np.log(d["RV_5d"].shift(-HORIZONTE).clip(lower=1e-8))
        columnas = columnas + ["Target"]

    d = d[columnas].replace([np.inf, -np.inf], np.nan)
    return d.dropna()


def cargar_dataset(fecha_inicio=FECHA_INICIO, fecha_fin=None):
    """Atajo: descarga y calcula características en un paso."""
    return ingenieria_caracteristicas(descargar_datos(fecha_inicio, fecha_fin))


# ==========================================
# 3. Utilidades de secuencias
# ==========================================
def construir_secuencias(X_escalado, filas, ventana=VENTANA):
    """
    Arma el tensor 3D (muestras, pasos, características) para las `filas` dadas.

    Devuelve también las filas efectivamente utilizadas: las primeras
    `ventana` posiciones del histórico no tienen contexto suficiente.
    """
    filas = np.asarray([i for i in filas if i >= ventana], dtype=int)
    if len(filas) == 0:
        raise ValueError("Ninguna fila tiene historial suficiente para la ventana.")
    tensor = np.stack([X_escalado[i - ventana : i] for i in filas])
    return tensor, filas

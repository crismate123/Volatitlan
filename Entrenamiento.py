import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import ta
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential


warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# 1. OBTENCION DE DATOS
# ==========================================
def obtener_datos_sp500(fecha_inicio="2015-01-01"):
    hoy = datetime.today()
    fecha_fin_yf = hoy + timedelta(days=1)  # yfinance usa end como fecha exclusiva
    str_fin = fecha_fin_yf.strftime("%Y-%m-%d")

    print(f"[*] Descargando datos del S&P 500 desde {fecha_inicio} hasta {str_fin}...")
    df = yf.download(
        "^GSPC",
        start=fecha_inicio,
        end=str_fin,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("yfinance no devolvio datos para ^GSPC.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    columnas_necesarias = {"Open", "High", "Low", "Close", "Volume"}
    columnas_faltantes = columnas_necesarias.difference(df.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(columnas_faltantes)}")

    return df


# ==========================================
# 2. INGENIERIA DE CARACTERISTICAS
# ==========================================
def ingenieria_caracteristicas(df):
    print("[*] Generando variables predictoras y variable objetivo...")
    datos = df.copy()

    datos["Log_Ret"] = np.log(datos["Close"] / datos["Close"].shift(1))

    datos["Vol_5d"] = datos["Log_Ret"].rolling(window=5).std()
    datos["Vol_20d"] = datos["Log_Ret"].rolling(window=20).std()
    datos["Vol_60d"] = datos["Log_Ret"].rolling(window=60).std()

    datos["RSI"] = ta.momentum.RSIIndicator(datos["Close"], window=14).rsi()
    # MACD y ATR se normalizan por el precio: en niveles crudos escalan con
    # el indice y caen fuera del rango de entrenamiento en produccion.
    datos["MACD"] = ta.trend.MACD(datos["Close"]).macd() / datos["Close"]
    datos["ATR"] = ta.volatility.AverageTrueRange(
        high=datos["High"],
        low=datos["Low"],
        close=datos["Close"],
        window=14,
    ).average_true_range() / datos["Close"]
    datos["Momento"] = ta.momentum.ROCIndicator(datos["Close"], window=10).roc()
    datos["Vol_Pct_Change"] = datos["Volume"].pct_change()
    # Sustitutos estacionarios de los precios/volumen en niveles:
    datos["Rango_Rel"] = (datos["High"] - datos["Low"]) / datos["Close"]
    datos["Vol_Rel"] = datos["Volume"] / datos["Volume"].rolling(window=20).mean()

    # RV(t+5) = sqrt(suma de retornos al cuadrado de los 5 dias siguientes)
    datos["Ret_Sq"] = datos["Log_Ret"] ** 2
    datos["RV_5d"] = np.sqrt(datos["Ret_Sq"].rolling(window=5).sum())
    datos["RV_22d"] = np.sqrt(datos["Ret_Sq"].rolling(window=22).sum())  # para HAR

    # Target en logaritmo: simetriza la distribucion (RV tiene cola pesada)
    # y hace que el error optimizado sea aproximadamente relativo.
    datos["Target"] = np.log(datos["RV_5d"].shift(-5).clip(lower=1e-8))

    datos = datos.drop(columns=["Ret_Sq"])  # RV_5d y RV_22d se conservan para benchmarks
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna()

    return datos


# ==========================================
# 3. CREACION DE SECUENCIAS (TENSOR 3D)
# ==========================================
def crear_secuencias(X_data, y_data, ventana=60):
    print(f"[*] Creando tensores de secuencias con ventana de {ventana} dias...")

    if len(X_data) != len(y_data):
        raise ValueError("X_data e y_data deben tener la misma cantidad de filas.")

    X, y = [], []
    for i in range(ventana, len(X_data)):
        X.append(X_data[i - ventana : i])
        y.append(y_data[i])

    return np.asarray(X), np.asarray(y)


# ==========================================
# 4. ARQUITECTURA Y ENTRENAMIENTO LSTM
# ==========================================
def construir_y_entrenar_lstm(X_train, y_train, X_val, y_val):
    print("[*] Construyendo y entrenando la red LSTM...")

    n_pasos = X_train.shape[1]
    n_caracteristicas = X_train.shape[2]

    modelo = Sequential(
        [
            Input(shape=(n_pasos, n_caracteristicas)),
            LSTM(64, activation="tanh", recurrent_dropout=0.2),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )

    # MAE sobre log(RV) escalado ~ optimizar el error relativo (afin al MAPE)
    modelo.compile(optimizer="adam", loss="mae", metrics=["mse"])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    historial = modelo.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    return modelo, historial


# ==========================================
# FLUJO PRINCIPAL (MAIN)
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    df_crudo = obtener_datos_sp500()
    df_features = ingenieria_caracteristicas(df_crudo)

    # Solo variables estacionarias: los precios/volumen en niveles crudos
    # rompen el escalado cuando el indice sube fuera del rango de train.
    columnas_x = [
        "Log_Ret",
        "Vol_5d",
        "Vol_20d",
        "Vol_60d",
        "RSI",
        "MACD",
        "ATR",
        "Momento",
        "Vol_Pct_Change",
        "Rango_Rel",
        "Vol_Rel",
    ]
    columna_y = "Target"

    df_modelo = df_features[columnas_x + [columna_y, "RV_5d", "RV_22d"]]

    ventana_dias = 60
    horizonte = 5  # dias futuros que usa el target
    split_index = int(len(df_modelo) * 0.8)

    if split_index <= ventana_dias + horizonte or len(df_modelo) - split_index <= horizonte:
        raise ValueError(
            "No hay suficientes datos para entrenar y validar con la ventana elegida."
        )

    # Purga anti-leakage: los ultimos `horizonte` targets del train usan
    # retornos que caen dentro del periodo de validacion, asi que se excluyen.
    X_train_df = df_modelo.iloc[: split_index - horizonte][columnas_x]
    y_train_df = df_modelo.iloc[: split_index - horizonte][[columna_y]]

    # La validacion necesita los ultimos dias de entrenamiento como contexto.
    X_val_df = df_modelo.iloc[split_index - ventana_dias :][columnas_x]
    y_val_df = df_modelo.iloc[split_index - ventana_dias :][[columna_y]]

    print("[*] Escalando datos...")
    # StandardScaler: robusto ante picos (un solo evento tipo COVID no
    # comprime el resto de la muestra como ocurre con MinMax).
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_df)
    y_train_scaled = scaler_y.fit_transform(y_train_df).ravel()

    X_val_scaled = scaler_x.transform(X_val_df)
    y_val_scaled = scaler_y.transform(y_val_df).ravel()

    X_train, y_train = crear_secuencias(
        X_train_scaled,
        y_train_scaled,
        ventana=ventana_dias,
    )
    X_val, y_val = crear_secuencias(
        X_val_scaled,
        y_val_scaled,
        ventana=ventana_dias,
    )

    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("No hay suficientes secuencias para entrenar o validar.")

    print(f"[*] Forma de X_train confirmada: {X_train.shape}")
    print(f"[*] Forma de X_val confirmada: {X_val.shape}")

    modelo, historial = construir_y_entrenar_lstm(X_train, y_train, X_val, y_val)

    # ==========================================
    # BENCHMARKS: LSTM vs PERSISTENCIA vs HAR-RV
    # ==========================================
    print("\n[*] Evaluando benchmarks en validacion (niveles de RV)...")

    # Reales y predicciones LSTM, desescalados y llevados de log a niveles
    rv_real = np.exp(scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel())
    rv_lstm = np.exp(scaler_y.inverse_transform(modelo.predict(X_val, verbose=0)).ravel())

    # Filas absolutas de df_modelo a las que corresponde cada secuencia de validacion
    n_val = len(y_val)
    filas_val = np.arange(split_index, split_index + n_val)

    # 1) Persistencia (naive): RV de la proxima semana = RV trailing actual
    rv_naive = df_modelo["RV_5d"].to_numpy()[filas_val]

    # 2) HAR-RV (log-log): regresion sobre RV diaria, semanal y mensual
    eps = 1e-8
    log_rv1 = np.log(np.abs(df_modelo["Log_Ret"].to_numpy()) + eps)
    log_rv5 = np.log(df_modelo["RV_5d"].to_numpy() + eps)
    log_rv22 = np.log(df_modelo["RV_22d"].to_numpy() + eps)
    y_log = df_modelo[columna_y].to_numpy()

    filas_train_har = np.arange(0, split_index - horizonte)
    A_train = np.column_stack([
        np.ones(len(filas_train_har)),
        log_rv1[filas_train_har],
        log_rv5[filas_train_har],
        log_rv22[filas_train_har],
    ])
    coef_har, *_ = np.linalg.lstsq(A_train, y_log[filas_train_har], rcond=None)
    A_val = np.column_stack([
        np.ones(n_val),
        log_rv1[filas_val],
        log_rv5[filas_val],
        log_rv22[filas_val],
    ])
    rv_har = np.exp(A_val @ coef_har)

    def _reportar(nombre, real, pred):
        err = real - pred
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / np.abs(real)) * 100)
        print(f"    {nombre:<24} RMSE={rmse:.5f}  MAE={mae:.5f}  MAPE={mape:5.1f}%")

    print("=" * 62)
    _reportar("LSTM", rv_real, rv_lstm)
    _reportar("Persistencia (naive)", rv_real, rv_naive)
    _reportar("HAR-RV (log-log)", rv_real, rv_har)
    print("=" * 62)
    print("    El LSTM debe batir a ambos benchmarks para validar la hipotesis.\n")

    # Guardar siempre junto a este script, sin importar el cwd
    carpeta = Path(__file__).resolve().parent
    modelo.save(carpeta / "modelo_lstm_sp500_volatilidad.keras")
    joblib.dump(scaler_x, carpeta / "scaler_x_sp500_volatilidad.pkl")
    joblib.dump(scaler_y, carpeta / "scaler_y_sp500_volatilidad.pkl")
    print(f"[*] Modelo y escaladores guardados correctamente en: {carpeta}")
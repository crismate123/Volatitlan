"""
Entrenamiento del pronosticador de volatilidad del S&P 500.

Arquitectura del sistema (no es solo una red):

    log RV(t+1..t+5)  =  HAR-X(t)  +  LSTM(residual)(t)

El HAR-X es una regresión lineal de 5 coeficientes sobre la volatilidad
realizada diaria, semanal y mensual más el VIX. Es el estándar de la
literatura y resulta muy difícil de batir. La red no compite con él: aprende
únicamente lo que el HAR-X deja sin explicar. Consecuencias:

  * el sistema nunca puede quedar materialmente peor que su baseline;
  * la red recibe un objetivo de varianza mucho menor;
  * si la red no aporta nada, se ve de inmediato en la tabla de benchmarks.

Metodología de evaluación:

  * Split temporal de tres vías 70/15/15 con purga de `HORIZONTE` filas en
    cada frontera (los targets solapados cruzan el corte).
  * La validación solo se usa para early stopping y para la corrección de
    Jensen. El test se toca UNA vez, al final.
  * Varias semillas promediadas: con un target solapado, la varianza entre
    corridas es grande y una sola realización no es un resultado.
  * Se reporta QLIKE además de los errores porcentuales. Es la pérdida
    estándar en volatilidad porque es robusta al ruido del proxy de RV; el
    MAPE está dominado por los periodos de calma, donde el denominador es
    pequeño.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from Caracteristicas import (
    COLUMNAS_HAR,
    COLUMNAS_X,
    FECHA_INICIO,
    HORIZONTE,
    VENTANA,
    cargar_dataset,
    construir_secuencias,
)
from Modelo import UNIDADES_LSTM, construir_lstm, ensamblar, n_parametros

CARPETA = Path(__file__).resolve().parent

ARCHIVO_MODELO = CARPETA / "modelo_lstm_sp500_volatilidad.keras"
ARCHIVO_SCALER_X = CARPETA / "scaler_x_sp500_volatilidad.pkl"
ARCHIVO_SCALER_Y = CARPETA / "scaler_y_sp500_volatilidad.pkl"
ARCHIVO_CONFIG = CARPETA / "config_modelo_sp500_volatilidad.json"

N_SEMILLAS = 5
EPOCAS = 120
BATCH = 64
PACIENCIA = 15


# ==========================================
# 1. Métricas
# ==========================================
def metricas(real, pred):
    """
    Métricas sobre niveles de RV. `real` y `pred` son desviaciones estándar.

    QLIKE se evalúa en varianza (RV²), que es su forma estándar. Es la única
    de estas métricas que penaliza correctamente subestimar la volatilidad en
    los episodios de estrés, que es cuando el error importa.
    """
    real = np.asarray(real, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-8, None)

    err = real - pred
    ape = np.abs(err) / np.abs(real)
    var_real, var_pred = real ** 2, pred ** 2

    log_real, log_pred = np.log(real), np.log(pred)
    ss_res = np.sum((log_real - log_pred) ** 2)
    ss_tot = np.sum((log_real - log_real.mean()) ** 2)

    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(ape) * 100),
        "MdAPE": float(np.median(ape) * 100),
        "QLIKE": float(np.mean(np.log(var_pred) + var_real / var_pred)),
        "R2_log": float(1 - ss_res / ss_tot),
    }


def imprimir_tabla(titulo, resultados):
    print(f"\n  {titulo}")
    print("  " + "-" * 86)
    print(f"  {'Modelo':<26}{'MAPE':>9}{'MdAPE':>9}{'RMSE':>11}{'MAE':>11}"
          f"{'QLIKE':>11}{'R2_log':>9}")
    print("  " + "-" * 86)
    for nombre, m in resultados.items():
        print(f"  {nombre:<26}{m['MAPE']:>8.1f}%{m['MdAPE']:>8.1f}%"
              f"{m['RMSE']:>11.5f}{m['MAE']:>11.5f}{m['QLIKE']:>11.4f}"
              f"{m['R2_log']:>9.3f}")
    print("  " + "-" * 86)


# ==========================================
# 2. Componente lineal (HAR / HAR-X)
# ==========================================
def ajustar_har(X, y):
    """Mínimos cuadrados con intercepto. Devuelve el vector de coeficientes."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predecir_har(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def a_niveles(log_pred, sigma2=0.0):
    """
    Convierte una predicción en log a niveles de RV.

    exp(mu) es la MEDIANA condicional, no la media: en logs el error es
    simétrico, pero al exponenciar la distribución se sesga. Sumar sigma²/2
    (corrección de Jensen) devuelve la media condicional, que es el estimador
    insesgado en niveles.
    """
    return np.exp(log_pred + sigma2 / 2.0)


# ==========================================
# FLUJO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("=" * 90)
    print(" ENTRENAMIENTO — VOLATILIDAD S&P 500  |  HAR-X + LSTM residual")
    print("=" * 90)

    # ---------- Datos ----------
    print(f"\n[1/6] Descargando y procesando datos desde {FECHA_INICIO}...")
    df = cargar_dataset(FECHA_INICIO)
    n = len(df)
    print(f"      {n:,} días hábiles  ({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})")
    print(f"      {len(COLUMNAS_X)} variables | ventana {VENTANA} | horizonte {HORIZONTE} días")

    # ---------- Split temporal con purga ----------
    i1, i2 = int(n * 0.70), int(n * 0.85)
    filas_train = np.arange(VENTANA, i1 - HORIZONTE)
    filas_val = np.arange(i1, i2 - HORIZONTE)
    filas_test = np.arange(i2, n)

    print(f"\n[2/6] Split temporal (purga de {HORIZONTE} días en cada frontera):")
    for nombre, filas in (("train", filas_train), ("val", filas_val), ("test", filas_test)):
        print(f"      {nombre:<6} {len(filas):>6,} filas   "
              f"{df.index[filas[0]]:%Y-%m-%d} → {df.index[filas[-1]]:%Y-%m-%d}")
    print(f"      muestra efectiva de train (targets no solapados): "
          f"~{len(filas_train) // HORIZONTE:,} observaciones")

    X_bruto = df[COLUMNAS_X].to_numpy()
    X_har = df[COLUMNAS_HAR].to_numpy()
    y_log = df["Target"].to_numpy()
    rv_real = np.exp(y_log)

    # ---------- Escalado (ajustado SOLO con train) ----------
    scaler_x = StandardScaler().fit(X_bruto[filas_train])
    X_esc = scaler_x.transform(X_bruto)

    # ---------- HAR-X: componente lineal ----------
    print("\n[3/6] Ajustando el componente lineal...")
    coef_har = ajustar_har(X_har[filas_train][:, :3], y_log[filas_train])   # sin VIX
    coef_harx = ajustar_har(X_har[filas_train], y_log[filas_train])         # con VIX
    har_todo = predecir_har(coef_har, X_har[:, :3])
    harx_todo = predecir_har(coef_harx, X_har)
    print("      HAR-X: intercepto=" + f"{coef_harx[0]:+.4f}  " +
          "  ".join(f"{c}={v:+.4f}" for c, v in zip(COLUMNAS_HAR, coef_harx[1:])))

    # La red aprende el residual del HAR-X, estandarizado
    residual = y_log - harx_todo
    scaler_y = StandardScaler().fit(residual[filas_train].reshape(-1, 1))
    resid_esc = scaler_y.transform(residual.reshape(-1, 1)).ravel()

    # ---------- Tensores ----------
    X_train, filas_train = construir_secuencias(X_esc, filas_train, VENTANA)
    X_val, filas_val = construir_secuencias(X_esc, filas_val, VENTANA)
    X_test, filas_test = construir_secuencias(X_esc, filas_test, VENTANA)
    y_train, y_val = resid_esc[filas_train], resid_esc[filas_val]

    # ---------- Entrenamiento multi-semilla ----------
    print(f"\n[4/6] Entrenando {N_SEMILLAS} redes LSTM({UNIDADES_LSTM}) sobre el residual...")
    redes, mape_semillas = [], []

    for k in range(N_SEMILLAS):
        red = construir_lstm(VENTANA, len(COLUMNAS_X), semilla=k,
                             nombre=f"lstm_semilla_{k}")
        red.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCAS, batch_size=BATCH, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=6, min_lr=1e-5, verbose=0),
            ],
        )
        redes.append(red)

        r = scaler_y.inverse_transform(red.predict(X_val, verbose=0)).ravel()
        m = metricas(rv_real[filas_val], a_niveles(harx_todo[filas_val] + r))
        mape_semillas.append(m["MAPE"])
        print(f"      semilla {k}:  MAPE(val) = {m['MAPE']:.1f}%   "
              f"R²log = {m['R2_log']:.3f}")

    print(f"      dispersión entre semillas: {np.mean(mape_semillas):.1f}% "
          f"± {np.std(mape_semillas):.1f} pp  →  se promedian en un ensamble")

    modelo = ensamblar(redes, VENTANA, len(COLUMNAS_X))
    print(f"      parámetros entrenables por red: {n_parametros(redes[0]):,}")

    # ---------- Corrección de Jensen, calibrada en validación ----------
    def log_pred_lstm(X_seq, filas):
        r = scaler_y.inverse_transform(modelo.predict(X_seq, verbose=0)).ravel()
        return harx_todo[filas] + r

    logp_val = log_pred_lstm(X_val, filas_val)
    sigma2 = float(np.var(y_log[filas_val] - logp_val))
    sigma2_har = float(np.var(y_log[filas_val] - har_todo[filas_val]))
    sigma2_harx = float(np.var(y_log[filas_val] - harx_todo[filas_val]))
    print(f"      sigma² del residual en validación: {sigma2:.4f} "
          f"(corrección de Jensen ×{np.exp(sigma2 / 2):.3f})")

    # ---------- Evaluación ----------
    print("\n[5/6] Evaluando sobre el conjunto de test (nunca visto)...")

    def comparativa(filas, X_seq):
        real = rv_real[filas]
        # Persistencia: la RV de la próxima semana será la de la semana pasada
        naive = df["RV_5d"].to_numpy()[filas]
        return {
            "Persistencia (naive)": metricas(real, naive),
            "HAR-RV": metricas(real, a_niveles(har_todo[filas], sigma2_har)),
            "HAR-X (con VIX)": metricas(real, a_niveles(harx_todo[filas], sigma2_harx)),
            "HAR-X + LSTM (final)": metricas(real, a_niveles(log_pred_lstm(X_seq, filas), sigma2)),
            # Mismo modelo sin corrección de Jensen: es la mediana condicional
            # y es lo que sirve la app. Se reporta porque el MAPE la prefiere
            # y porque sigma² está calibrado en un régimen más volátil.
            "HAR-X + LSTM (mediana)": metricas(real, a_niveles(log_pred_lstm(X_seq, filas))),
        }

    res_val = comparativa(filas_val, X_val)
    res_test = comparativa(filas_test, X_test)
    imprimir_tabla("VALIDACIÓN (usada para early stopping — optimista)", res_val)
    imprimir_tabla("TEST (nunca visto — este es el número honesto)", res_test)

    mejora = ((res_test["HAR-X (con VIX)"]["QLIKE"] - res_test["HAR-X + LSTM (final)"]["QLIKE"])
              / abs(res_test["HAR-X (con VIX)"]["QLIKE"]) * 100)
    print(f"\n  Aporte de la red sobre su propio baseline (QLIKE en test): {mejora:+.2f}%")
    if mejora <= 0:
        print("  ⚠️  La red NO está aportando sobre el HAR-X. Usa el HAR-X solo.")

    # ---------- Diagnóstico del MAPE por régimen ----------
    real_test = rv_real[filas_test]
    pred_test = a_niveles(log_pred_lstm(X_test, filas_test), sigma2)
    quintiles = np.quantile(real_test, [0.2, 0.4, 0.6, 0.8])
    grupo = np.digitize(real_test, quintiles)
    print("\n  MAPE por quintil de volatilidad realizada (test):")
    etiquetas = ["Q1 (más calmado)", "Q2", "Q3", "Q4", "Q5 (más volátil)"]
    for g, etq in enumerate(etiquetas):
        sel = grupo == g
        if sel.sum():
            ape = np.abs(real_test[sel] - pred_test[sel]) / real_test[sel]
            print(f"      {etq:<20} RV media={real_test[sel].mean():.4f}   "
                  f"MAPE={ape.mean() * 100:5.1f}%   n={sel.sum()}")
    print("      (el MAPE alto se concentra en los quintiles calmados: el "
          "denominador es pequeño)")

    # ---------- Persistencia de artefactos ----------
    print("\n[6/6] Guardando artefactos...")
    modelo.save(ARCHIVO_MODELO)
    joblib.dump(scaler_x, ARCHIVO_SCALER_X)
    joblib.dump(scaler_y, ARCHIVO_SCALER_Y)

    config = {
        "version": 2,
        "fecha_entrenamiento": datetime.today().strftime("%Y-%m-%d"),
        "fecha_inicio_datos": str(df.index[0].date()),
        "fecha_fin_datos": str(df.index[-1].date()),
        "ventana": VENTANA,
        "horizonte": HORIZONTE,
        "columnas_x": COLUMNAS_X,
        "columnas_har": COLUMNAS_HAR,
        "coef_harx": coef_harx.tolist(),
        "coef_har": coef_har.tolist(),
        "sigma2": sigma2,
        "n_semillas": N_SEMILLAS,
        "metricas_test": res_test,
        "metricas_val": res_val,
    }
    ARCHIVO_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    for archivo in (ARCHIVO_MODELO, ARCHIVO_SCALER_X, ARCHIVO_SCALER_Y, ARCHIVO_CONFIG):
        print(f"      {archivo.name}")
    print("\n" + "=" * 90)

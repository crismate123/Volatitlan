"""
Modelos econométricos de referencia para la selección de modelos.

Implementa los tres benchmarks clásicos de la literatura de volatilidad:

- HAR-RV (Corsi, 2009): regresión lineal de la RV futura sobre sus componentes
  heterogéneos diario, semanal y mensual. A diferencia del HAR-X interno del
  pipeline de la LSTM, aquí NO se incluye el VIX: es el HAR-RV canónico.
- GARCH(1,1) (Bollerslev, 1986): varianza condicional de los retornos diarios
  con persistencia simétrica.
- EGARCH(1,1) (Nelson, 1991): extensión en logaritmos con término asimétrico,
  que captura el efecto apalancamiento (los retornos negativos elevan más la
  volatilidad que los positivos).

Protocolo anti look-ahead: los parámetros de los tres modelos se estiman
EXCLUSIVAMENTE con datos anteriores a `fecha_corte` y quedan congelados; las
predicciones posteriores al corte solo usan la información disponible hasta
cada día t (pseudo out-of-sample). Es el mismo protocolo con el que se evalúa
la LSTM, cuyo entrenamiento también terminó antes del periodo evaluado.

Objetivo común: todos pronostican lo mismo que la red, la volatilidad
realizada de los próximos 5 días hábiles, RV(t+1..t+5). Para los GARCH la
varianza semanal es la suma de las cinco varianzas diarias pronosticadas
(E[Σr²] = Σ varianzas condicionales).
"""

import numpy as np
import pandas as pd
from arch import arch_model

HORIZONTE = 5
_EPS = 1e-8

# Los retornos se escalan ×100 para el optimizador de `arch` (recomendación
# de la propia librería: verosimilitudes mal escaladas no convergen bien).
_ESCALA = 100.0


# ==========================================
# 1. HAR-RV (Corsi, 2009)
# ==========================================
def predecir_har_rv(feats, fecha_corte):
    """
    HAR-RV estimado por OLS en logaritmos.

    log RV(t+1..t+5) ~ const + log RV_1d + log RV_5d + log RV_22d

    Se estima solo con filas hasta `fecha_corte` y devuelve la predicción en
    niveles (exp del log-pronóstico, la mediana condicional: el mismo criterio
    que usa la LSTM, ver `Prediccion.a_niveles`).
    """
    cols = ["log_RV_1d", "log_RV_5d", "log_RV_22d"]
    X = np.column_stack([np.ones(len(feats)), feats[cols].to_numpy()])
    y = np.log(feats["RV_5d"].shift(-HORIZONTE).clip(lower=_EPS)).to_numpy()

    en_train = (feats.index <= fecha_corte) & ~np.isnan(y)
    if int(en_train.sum()) < 100:
        raise ValueError("Muestra insuficiente para estimar el HAR-RV.")

    coef, *_ = np.linalg.lstsq(X[en_train], y[en_train], rcond=None)
    return pd.Series(np.exp(X @ coef), index=feats.index, name="HAR-RV")


# ==========================================
# 2. GARCH(1,1) y EGARCH(1,1)
# ==========================================
def _pronostico_arch(log_ret, fecha_corte, vol, o=0):
    """
    Ajusta un modelo de la familia ARCH hasta `fecha_corte` y pronostica la RV
    semanal para TODAS las fechas desde el corte con parámetros congelados.

    Devuelve una serie indexada por la fecha de origen del pronóstico: el
    valor en t es la RV pronosticada para t+1..t+5, igual que el target.
    """
    ret = (log_ret.dropna() * _ESCALA).astype(float)
    am = arch_model(ret, mean="Constant", vol=vol, p=1, o=o, q=1,
                    dist="normal", rescale=False)
    res = am.fit(last_obs=fecha_corte, disp="off", show_warning=False)

    if vol == "EGARCH":
        # El EGARCH no tiene pronóstico multi-paso en forma cerrada (la
        # no-linealidad del log impide iterar la esperanza): simulación con
        # semilla fija para que el resultado sea reproducible entre corridas.
        rs = np.random.RandomState(42)
        fc = res.forecast(horizon=HORIZONTE, start=fecha_corte, reindex=False,
                          method="simulation", simulations=1000,
                          rng=lambda size: rs.standard_normal(size))
    else:
        fc = res.forecast(horizon=HORIZONTE, start=fecha_corte, reindex=False)

    # Varianza de la semana = suma de las 5 varianzas diarias (en %²)
    var_semana = fc.variance.sum(axis=1)
    return np.sqrt(var_semana) / _ESCALA


def predecir_garch(log_ret, fecha_corte):
    """GARCH(1,1) con media constante e innovaciones normales."""
    return _pronostico_arch(log_ret, fecha_corte, "GARCH").rename("GARCH(1,1)")


def predecir_egarch(log_ret, fecha_corte):
    """EGARCH(1,1) con término asimétrico (o=1)."""
    return _pronostico_arch(log_ret, fecha_corte, "EGARCH", o=1).rename("EGARCH")


# ==========================================
# BLOQUE DE PRUEBA / EJECUCIÓN AISLADA
# ==========================================
if __name__ == "__main__":
    from Caracteristicas import descargar_datos, ingenieria_caracteristicas

    print("--- PRUEBA DE BENCHMARKS ---")
    df = descargar_datos("2005-01-01")
    feats = ingenieria_caracteristicas(df, con_target=False)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    corte = feats.index.max() - pd.Timedelta(days=730)

    har = predecir_har_rv(feats, corte)
    garch = predecir_garch(log_ret, corte)
    egarch = predecir_egarch(log_ret, corte)

    print(f"[+] Corte de estimación: {corte:%Y-%m-%d}")
    for s in (har[har.index > corte], garch, egarch):
        print(f"[+] {s.name}: {len(s)} pronósticos, "
              f"último ({s.index[-1]:%Y-%m-%d}) = {s.iloc[-1]:.4f}")
    print("[+] Benchmarks validados.")

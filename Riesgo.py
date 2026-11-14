"""
Valor en Riesgo (VaR) y VaR condicional (CVaR) semanales a partir del
pronóstico de volatilidad del modelo.

Método: simulación histórica filtrada (FHS, Barone-Adesi et al., 1999).

1. Para cada día t fuera de la muestra de entrenamiento se estandariza el
   retorno de los 5 días siguientes con la volatilidad que el modelo
   pronosticó en t:

       z_t = R(t+1..t+5) / RV_pronosticada(t)

2. La distribución empírica de z recoge lo que una normal no ve: colas
   pesadas, asimetría negativa del índice y cualquier sesgo sistemático del
   pronóstico (si el modelo subestima, los z son más grandes y el VaR crece
   solo).

3. Para la semana que se quiere medir, el retorno posible es z · RV
   pronosticada, y la pérdida sobre un monto M es M · (1 - e^{z·RV}).

   VaR_α  = pérdida en el cuantil 1-α de z  (la pérdida que solo se supera
            con probabilidad 1-α)
   CVaR_α = pérdida promedio en las semanas que superan el VaR
            (cuánto se pierde, en promedio, cuando se pierde "de más")

Solo se usan días posteriores al entrenamiento: con días de entrenamiento los
residuales son artificialmente pequeños (el modelo ya los vio) y el VaR
saldría subestimado.

Inversionista en pesos sin cobertura cambiaria
----------------------------------------------
Quien invierte pesos en el S&P 500 compra dólares, compra el índice y al
final vende dólares. Su retorno logarítmico es la suma de los dos:

    R_MXN = R_S&P + R_USDMXN

La volatilidad semanal de esa suma combina el pronóstico del modelo para el
S&P con una estimación EWMA (RiskMetrics, λ = 0.94) de la volatilidad del
tipo de cambio y de su correlación con el índice:

    σ_MXN² = σ_S&P² + σ_FX² + 2 ρ σ_S&P σ_FX

La correlación suele ser NEGATIVA: en episodios de aversión al riesgo el S&P
cae y el peso se deprecia (sube USD/MXN), así que el tipo de cambio amortigua
parte de la pérdida medida en pesos. El resto del método no cambia: se
estandarizan los retornos combinados con σ_MXN y se aplica FHS.

Este módulo no depende de Streamlit para poder probarse por separado.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

HORIZONTE = 5


def retorno_adelantado(close, horizonte=HORIZONTE):
    """Retorno logarítmico de t a t+horizonte sesiones: log(C_{t+h} / C_t)."""
    return np.log(close.shift(-horizonte) / close)


LAMBDA_EWMA = 0.94  # RiskMetrics para datos diarios


def alinear_tipo_cambio(fx):
    """
    Alinea el USD/MXN de Yahoo (`MXN=X`) con el cierre de Nueva York.

    La barra diaria que Yahoo fecha en D+1 refleja el mercado al cierre de
    la sesión estadounidense de D: medido desde 2016, la correlación diaria
    entre el retorno del S&P de D y el del USD/MXN fechado en D+1 es -0.33,
    frente a -0.09 con la barra del mismo día (y -0.39 a 5 días, donde el
    desfase ya no importa). Sin esta corrección la correlación EWMA sale con
    el signo equivocado y la volatilidad en pesos se sobreestima.

    La última sesión todavía no tiene barra siguiente: se usa la cotización
    más reciente disponible.
    """
    alineado = fx.shift(-1)
    if len(fx):
        alineado.iloc[-1] = fx.iloc[-1]
    return alineado


def volatilidad_combinada(sigma_sp, close_sp, fx, horizonte=HORIZONTE, lam=LAMBDA_EWMA):
    """
    Volatilidad a `horizonte` días del retorno en pesos (S&P + USD/MXN).

    sigma_sp: pronóstico del modelo (RV a 5 días del S&P), indexado por fecha.
    close_sp, fx: cierres diarios del índice y del tipo de cambio (pesos por
    dólar) sobre el mismo calendario de sesiones.

    Devuelve un DataFrame alineado a `sigma_sp` con la volatilidad del tipo
    de cambio, la correlación EWMA y la volatilidad combinada. Las EWMA en t
    solo usan información hasta el cierre de t.
    """
    r_sp = np.log(close_sp / close_sp.shift(1))
    r_fx = np.log(fx / fx.shift(1))
    datos = pd.DataFrame({"sp": r_sp, "fx": r_fx}).dropna()
    alfa = 1.0 - lam
    var_sp = (datos["sp"] ** 2).ewm(alpha=alfa, adjust=False).mean()
    var_fx = (datos["fx"] ** 2).ewm(alpha=alfa, adjust=False).mean()
    cov = (datos["sp"] * datos["fx"]).ewm(alpha=alfa, adjust=False).mean()
    rho = (cov / np.sqrt(var_sp * var_fx)).clip(-0.99, 0.99)

    sigma_fx = np.sqrt(var_fx * horizonte).reindex(sigma_sp.index)
    rho = rho.reindex(sigma_sp.index)
    sigma_c = np.sqrt(sigma_sp ** 2 + sigma_fx ** 2 + 2.0 * rho * sigma_sp * sigma_fx)
    return pd.DataFrame({"sigma_fx": sigma_fx, "rho": rho, "sigma_mxn": sigma_c})


def residuales_estandarizados(pronostico, retorno):
    """z_t = R(t+1..t+5) / RV pronosticada(t), solo donde ambos existen."""
    z = (retorno / pronostico).replace([np.inf, -np.inf], np.nan)
    return z.dropna()


def _perdida(monto, retorno_log):
    """Pérdida en dinero (positiva = se pierde) de un retorno logarítmico."""
    return monto * (1.0 - np.exp(retorno_log))


def var_cvar_fhs(z, sigma, nivel, monto):
    """
    VaR y CVaR por simulación histórica filtrada.

    z: residuales estandarizados históricos; sigma: RV pronosticada para la
    semana; nivel: confianza (0.95); monto: capital invertido.
    """
    z = np.asarray(z, dtype=float)
    q = float(np.quantile(z, 1.0 - nivel))
    cola = z[z <= q]
    return {
        "VaR": float(_perdida(monto, q * sigma)),
        "CVaR": float(np.mean(_perdida(monto, cola * sigma))),
        "cuantil_z": q,
    }


def var_cvar_normal(sigma, nivel, monto):
    """
    Referencia paramétrica: retorno logarítmico ~ N(0, sigma²).

    CVaR en forma cerrada para la pérdida M(1 - e^X), X ~ N(0, σ²):
        E[e^X | X <= q] = e^{σ²/2} · Φ(q/σ - σ) / Φ(q/σ)
    """
    nd = NormalDist()
    q = nd.inv_cdf(1.0 - nivel) * sigma
    p = 1.0 - nivel
    media_cola = math.exp(sigma ** 2 / 2.0) * nd.cdf(q / sigma - sigma) / p
    return {"VaR": float(_perdida(monto, q)), "CVaR": float(monto * (1.0 - media_cola))}


def prueba_kupiec(n, excesos, nivel):
    """
    Prueba de cobertura incondicional de Kupiec (1995).

    H0: la frecuencia de semanas que superan el VaR es 1-α. Devuelve el
    p-valor (chi-cuadrada con 1 g.l.). p < 0.05 indica que el VaR está mal
    calibrado (demasiados o muy pocos excesos).
    """
    p = 1.0 - nivel
    if n == 0:
        return float("nan")
    x = int(excesos)
    log_h0 = (n - x) * math.log(1 - p) + x * math.log(p)
    phat = x / n
    log_h1 = 0.0
    if 0 < x < n:
        log_h1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
    lr = max(-2.0 * (log_h0 - log_h1), 0.0)
    return float(math.erfc(math.sqrt(lr / 2.0)))  # supervivencia chi²(1)


def backtest_semanal(pronostico, retorno, nivel, monto, anios=2, min_calibracion=250):
    """
    VaR semanal sin solapamiento sobre los últimos `anios`.

    Cada semana se mide desde el último cierre de la semana previa. El VaR de
    esa semana se calcula SOLO con residuales cuyo retorno ya se conocía en
    ese momento (ventana expansiva), así que no hay look-ahead.
    """
    z_todos = residuales_estandarizados(pronostico, retorno)
    datos = pd.DataFrame({"sigma": pronostico, "ret": retorno}).dropna()
    corte = datos.index.max() - pd.Timedelta(days=int(anios * 365))
    datos = datos[datos.index > corte]

    # Último cierre de cada semana calendario
    bases = datos.groupby(datos.index.to_period("W-FRI")).tail(1)

    filas = []
    for fecha, fila in bases.iterrows():
        # Un residual de t solo se conoce en t+5 sesiones: 10 días naturales
        # de margen garantizan que ya estaba observado en `fecha`.
        z_prev = z_todos[z_todos.index <= fecha - pd.Timedelta(days=10)]
        if len(z_prev) < min_calibracion:
            continue
        r = var_cvar_fhs(z_prev, fila["sigma"], nivel, monto)
        perdida = float(_perdida(monto, fila["ret"]))
        filas.append({
            "Fecha": fecha,
            "PyG": -perdida,
            "VaR": r["VaR"],
            "CVaR": r["CVaR"],
            "Exceso": perdida > r["VaR"],
        })
    return pd.DataFrame(filas).set_index("Fecha") if filas else pd.DataFrame()

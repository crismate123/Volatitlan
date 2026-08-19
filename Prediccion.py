"""
Pipeline de inferencia: HAR-X + LSTM residual.

Este módulo NO reimplementa la ingeniería de características: la importa de
`Caracteristicas.py`, el mismo módulo que usa el entrenamiento. Es la garantía
estructural de que producción y entrenamiento no pueden divergir.

Toda la configuración del modelo (ventana, columnas, coeficientes del HAR-X,
corrección de Jensen) vive en el JSON que produce el entrenamiento, de modo
que reentrenar con otros hiperparámetros no exige tocar este archivo.
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from Caracteristicas import (
    construir_secuencias,
    descargar_datos,
    ingenieria_caracteristicas,
)

warnings.filterwarnings("ignore")

_CARPETA = Path(__file__).resolve().parent

# Días naturales que se descargan para inferencia. Se necesitan ~66 sesiones
# de calentamiento (RV_66d) más la ventana del modelo; 400 días naturales
# (~275 hábiles) dejan margen suficiente.
DIAS_DESCARGA = 400


class PredictorVolatilidad:
    """Carga los artefactos entrenados y ejecuta el pronóstico de RV a 5 días."""

    def __init__(
        self,
        ruta_modelo=_CARPETA / "modelo_lstm_sp500_volatilidad.keras",
        ruta_scaler_x=_CARPETA / "scaler_x_sp500_volatilidad.pkl",
        ruta_scaler_y=_CARPETA / "scaler_y_sp500_volatilidad.pkl",
        ruta_config=_CARPETA / "config_modelo_sp500_volatilidad.json",
    ):
        if not Path(ruta_config).exists():
            raise FileNotFoundError(
                f"Falta {Path(ruta_config).name}. Ejecuta Entrenamiento.py para "
                "generar los artefactos del modelo."
            )

        self.config = json.loads(Path(ruta_config).read_text(encoding="utf-8"))
        self.modelo = load_model(ruta_modelo)
        self.scaler_x = joblib.load(ruta_scaler_x)
        self.scaler_y = joblib.load(ruta_scaler_y)

        self.ventana = int(self.config["ventana"])
        self.horizonte = int(self.config["horizonte"])
        self.columnas_features = list(self.config["columnas_x"])
        self.columnas_har = list(self.config["columnas_har"])
        self.coef_harx = np.asarray(self.config["coef_harx"], dtype=float)
        self.sigma2 = float(self.config["sigma2"])

    # ------------------------------------------------------------------
    # Datos
    # ------------------------------------------------------------------
    def obtener_datos_recientes(self, dias=DIAS_DESCARGA):
        """Descarga ^GSPC y ^VIX suficientes para formar una ventana limpia."""
        hoy = datetime.today()
        return descargar_datos(hoy - timedelta(days=dias), hoy + timedelta(days=1))

    def preparar(self, df):
        """
        Calcula las características sobre un frame con OHLCV + VIX.

        `con_target=False`: en producción la RV de la próxima semana todavía no
        existe, y exigirla borraría justamente las filas que se quieren predecir.
        """
        return ingenieria_caracteristicas(df, con_target=False)

    def matrices(self, feats):
        """Devuelve (X escalado, predicción del HAR-X en log) para todo el frame."""
        X_esc = self.scaler_x.transform(feats[self.columnas_features].to_numpy())
        X_har = feats[self.columnas_har].to_numpy()
        harx = np.column_stack([np.ones(len(X_har)), X_har]) @ self.coef_harx
        return X_esc, harx

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------
    def predecir_log(self, X_esc, harx, filas):
        """
        Pronóstico en logaritmo para las filas indicadas.

        Devuelve (log_pred, filas_usadas): las filas sin historial suficiente
        para llenar la ventana se descartan.
        """
        tensor, filas_ok = construir_secuencias(X_esc, filas, self.ventana)
        resid = self.scaler_y.inverse_transform(
            self.modelo.predict(tensor, verbose=0)
        ).ravel()
        return harx[filas_ok] + resid, filas_ok

    def a_niveles(self, log_pred, corregir_sesgo=False):
        """
        Pasa de log a niveles de RV.

        `exp(mu)` es la MEDIANA condicional; sumar sigma²/2 (corrección de
        Jensen) da la MEDIA. El default es la mediana, y no por descuido:

        El sigma² se calibra en validación (2015-2021), un tramo que contiene
        el shock de COVID y por tanto residuales muy dispersos. Aplicado al
        régimen reciente, ese factor (×1.10) sobre-corrige: medido sobre los
        últimos 2 años, la mediana sobreestima la RV en +9.4% en promedio y la
        versión "corregida" lo hace en +20.7%. La corrección empeora el sesgo
        en lugar de eliminarlo.

        Además, la mediana es el estimador que minimiza el error relativo, que
        es lo que reportan el MAPE y el MdAPE de la aplicación. QLIKE y RMSE
        prefieren marginalmente la media; si se reentrena con un sigma²
        recalibrado sobre un periodo homogéneo, conviene reevaluar este default.
        """
        return np.exp(log_pred + (self.sigma2 / 2.0 if corregir_sesgo else 0.0))

    def predecir_en_filas(self, X_esc, harx, filas, corregir_sesgo=False):
        """Atajo: log → niveles. Devuelve (RV en niveles, filas usadas)."""
        log_pred, filas_ok = self.predecir_log(X_esc, harx, filas)
        return self.a_niveles(log_pred, corregir_sesgo), filas_ok

    def predecir_futuro(self, corregir_sesgo=False):
        """RV esperada para los próximos 5 días hábiles, a partir de hoy."""
        feats = self.preparar(self.obtener_datos_recientes())
        if len(feats) < self.ventana:
            raise ValueError(
                f"Datos insuficientes: {len(feats)} filas limpias para una "
                f"ventana de {self.ventana}."
            )
        X_esc, harx = self.matrices(feats)
        niveles, _ = self.predecir_en_filas(
            X_esc, harx, [len(feats) - 1], corregir_sesgo
        )
        return float(niveles[0])


# ==========================================
# EJECUCIÓN DEL SCRIPT
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(" INFERENCIA — VOLATILIDAD S&P 500 (HAR-X + LSTM residual)")
    print("=" * 60)

    try:
        predictor = PredictorVolatilidad()
        cfg = predictor.config
        print(f"[*] Modelo entrenado el {cfg['fecha_entrenamiento']} con datos "
              f"{cfg['fecha_inicio_datos']} → {cfg['fecha_fin_datos']}")

        rv = predictor.predecir_futuro()
        rv_anual = rv * np.sqrt(252.0 / predictor.horizonte)

        print("\n" + "=" * 60)
        print(f" RV estimada (próximos {predictor.horizonte} días): {rv:.6f}")
        print(f" Equivalente anualizado:                    {rv_anual * 100:.2f}%")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[!] Error durante la ejecución de la predicción: {e}")

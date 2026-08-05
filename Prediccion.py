import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Carpeta donde vive este script (los artefactos del modelo se buscan aquí)
_CARPETA = Path(__file__).resolve().parent
import joblib
from tensorflow.keras.models import load_model
import ta
import warnings

# Ignorar warnings de yfinance o pandas para mantener la consola limpia
warnings.filterwarnings('ignore')

class PredictorVolatilidad:
    """
    Clase encargada de ejecutar el pipeline de inferencia:
    descarga de datos recientes, preprocesamiento, escalado y predicción.
    """
    def __init__(
        self,
        ruta_modelo=_CARPETA / "modelo_lstm_sp500_volatilidad.keras",
        ruta_scaler_x=_CARPETA / "scaler_x_sp500_volatilidad.pkl",
        ruta_scaler_y=_CARPETA / "scaler_y_sp500_volatilidad.pkl",
    ):
        print("[*] Cargando modelo y escaladores desde el disco...")
        self.modelo = load_model(ruta_modelo)
        self.scaler_x = joblib.load(ruta_scaler_x)
        self.scaler_y = joblib.load(ruta_scaler_y)
        
        # Columnas exactas que espera el modelo, EN EL MISMO ORDEN que en
        # el entrenamiento (solo variables estacionarias)
        self.columnas_features = [
            'Log_Ret', 'Vol_5d', 'Vol_20d', 'Vol_60d', 'RSI', 'MACD',
            'ATR', 'Momento', 'Vol_Pct_Change', 'Rango_Rel', 'Vol_Rel'
        ]

    def obtener_datos_recientes(self):
        """
        Descarga los últimos ~250 días naturales. Se necesita buffer porque
        Vol_60d consume 60 registros en NaN y la ventana del modelo requiere
        otros 60 días limpios (~120 días hábiles en total).
        """
        hoy = datetime.today()
        fecha_inicio = hoy - timedelta(days=250)
        fecha_fin = hoy + timedelta(days=1)  # end es exclusivo en yfinance

        print(f"[*] Descargando datos recientes desde {fecha_inicio.strftime('%Y-%m-%d')}...")
        df = yf.download(
            "^GSPC",
            start=fecha_inicio.strftime('%Y-%m-%d'),
            end=fecha_fin.strftime('%Y-%m-%d'),
            auto_adjust=False,  # igual que en el entrenamiento
            progress=False,
        )

        if df.empty:
            raise ValueError("yfinance no devolvió datos para ^GSPC.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df

    def ingenieria_caracteristicas_inferencia(self, df):
        """
        Replica la ingeniería de características EXACTA del entrenamiento, 
        pero SIN calcular el Target futuro (ya que es lo que queremos predecir).
        """
        print("[*] Aplicando ingeniería de características...")
        datos = df.copy()
        
        datos['Log_Ret'] = np.log(datos['Close'] / datos['Close'].shift(1))
        datos['Vol_5d'] = datos['Log_Ret'].rolling(window=5).std()
        datos['Vol_20d'] = datos['Log_Ret'].rolling(window=20).std()
        datos['Vol_60d'] = datos['Log_Ret'].rolling(window=60).std()
        
        datos['RSI'] = ta.momentum.RSIIndicator(datos['Close'], window=14).rsi()
        # MACD y ATR normalizados por el precio (igual que en el entrenamiento)
        datos['MACD'] = ta.trend.MACD(datos['Close']).macd() / datos['Close']
        datos['ATR'] = ta.volatility.AverageTrueRange(datos['High'], datos['Low'], datos['Close'], window=14).average_true_range() / datos['Close']
        datos['Momento'] = ta.momentum.ROCIndicator(datos['Close'], window=10).roc()
        datos['Vol_Pct_Change'] = datos['Volume'].pct_change()
        datos['Rango_Rel'] = (datos['High'] - datos['Low']) / datos['Close']
        datos['Vol_Rel'] = datos['Volume'] / datos['Volume'].rolling(window=20).mean()

        # Eliminamos infinitos (ej. Vol_Pct_Change con volumen 0) y NaNs de los rolling windows
        datos = datos.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Retornamos solo las columnas que necesita el modelo
        return datos[self.columnas_features]

    def predecir_futuro(self):
        """
        Ejecuta el flujo completo para predecir la volatilidad de los próximos 5 días.
        """
        # 1. Obtener y preparar datos
        df_crudo = self.obtener_datos_recientes()
        df_features = self.ingenieria_caracteristicas_inferencia(df_crudo)
        
        # Validar que tengamos al menos 60 días de datos limpios
        if len(df_features) < 60:
            raise ValueError("No hay suficientes datos limpios para formar una ventana de 60 días.")
            
        # 2. Extraer ÚNICAMENTE los últimos 60 días (la ventana actual)
        ultimos_60_dias = df_features.tail(60)

        # 3. Escalar las features con el mismo scaler_x del entrenamiento
        X_inferencia = self.scaler_x.transform(ultimos_60_dias)

        # 4. Ajustar dimensión para LSTM -> (Muestras, Pasos_temporales, Características) -> (1, 60, 14)
        X_tensor = np.expand_dims(X_inferencia, axis=0)

        # 5. Predicción
        prediccion_escalada = self.modelo.predict(X_tensor, verbose=0)

        # 6. Desescalar y deshacer el logaritmo para obtener RV en niveles
        prediccion_real = self.desescalar_prediccion(prediccion_escalada)[0, 0]

        return float(prediccion_real)

    def desescalar_prediccion(self, prediccion_escalada):
        """
        Convierte salidas escaladas del modelo a RV en niveles:
        inverso del StandardScaler + exponencial (el target se entrenó en log).
        """
        return np.exp(self.scaler_y.inverse_transform(prediccion_escalada))


# ==========================================
# EJECUCIÓN DEL SCRIPT
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print(" INICIANDO PIPELINE DE INFERENCIA S&P 500 (LSTM)")
    print("==================================================")
    
    try:
        # Instanciar la clase (cargará los archivos .keras y .pkl automáticamente)
        predictor = PredictorVolatilidad()
        
        # Ejecutar la predicción
        volatilidad_esperada = predictor.predecir_futuro()
        
        print("\n" + "="*50)
        print(f" PREDICCIÓN EXITOSA")
        print(f" Volatilidad Realizada estimada (próximos 5 días): {volatilidad_esperada:.6f}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[!] Error durante la ejecución de la predicción: {e}")
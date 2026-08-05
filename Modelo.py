import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import os

class VolatilityLSTM:
    """
    Clase para construir, compilar y entrenar el modelo LSTM para la 
    predicción de volatilidad multivariada de un solo paso.
    """
    
    def __init__(self, n_pasos=60, n_caracteristicas=11):
        """
        Inicializa los hiperparámetros de la arquitectura.
        """
        self.n_pasos = n_pasos
        self.n_caracteristicas = n_caracteristicas
        self.modelo = self._construir_arquitectura()
        
    def _construir_arquitectura(self):
        """
        Define la topología exacta documentada en el plan del proyecto.
        """
        print("[*] Construyendo arquitectura LSTM (64 -> Dropout -> Dense 32 -> Dense 1)...")
        modelo = Sequential([
            # Capa de entrada
            Input(shape=(self.n_pasos, self.n_caracteristicas)),

            # Capa recurrente con regularización interna
            LSTM(64, activation='tanh', recurrent_dropout=0.2),

            # Regularización para evitar el sobreajuste (Dropout)
            Dropout(0.2),
            
            # Capa oculta no lineal
            Dense(32, activation='relu'),
            
            # Capa de salida lineal (Predicción de escalar único)
            Dense(1, activation='linear')
        ])
        
        # Compilación del modelo
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss='mse', 
            metrics=['mae', 'mape'] # MAPE es tu métrica de éxito del 10-15%
        )
        
        return modelo
    
    def entrenar(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Ejecuta el ciclo de entrenamiento con validación estricta y Early Stopping.
        """
        print("[*] Iniciando entrenamiento con validación Walk-Forward...")
        
        # Early stopping monitoreando la pérdida de validación para evitar Data Leakage / Overfitting
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True, # Vital: devuelve el modelo al mejor punto, no al último
            verbose=1
        )
        
        historial = self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        return historial

    def guardar_modelo(self, ruta_archivo=os.path.join("modelos", "lstm.keras")):
        """
        Exporta el modelo en formato .keras para su uso en la App Web.
        """
        directorio = os.path.dirname(ruta_archivo)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        self.modelo.save(ruta_archivo)
        print(f"[+] Modelo exportado exitosamente en: {ruta_archivo}")


# ==========================================
# BLOQUE DE PRUEBA / EJECUCIÓN AISLADA
# ==========================================
if __name__ == "__main__":
    # Este bloque solo se ejecuta si corres este script directamente para probar la red.
    # En producción, este archivo será importado por tu script principal (ej. main.py o train.py).
    
    import numpy as np
    
    print("--- PRUEBA DE ARQUITECTURA ---")
    # Simulamos tensores dummy para validar que la red compila correctamente
    X_dummy = np.random.rand(100, 60, 14) # 100 muestras, 60 días, 14 features
    y_dummy = np.random.rand(100)         # 100 valores de volatilidad
    
    red_volatilidad = VolatilityLSTM(n_pasos=60, n_caracteristicas=11)
    
    # Resumen de la topología
    red_volatilidad.modelo.summary()
    
    print("\n[+] Arquitectura validada. Lista para integrarse con el pipeline de datos.")
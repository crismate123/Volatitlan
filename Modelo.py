"""
Arquitectura de la red para el pronóstico de volatilidad del S&P 500.

Este módulo es la ÚNICA fuente de verdad de la topología: `Entrenamiento.py`
lo importa, de modo que no puede haber divergencia entre lo que se documenta
aquí y lo que realmente se entrena.

Nota de diseño: la red NO predice la volatilidad. Predice el *residual* de un
HAR-X (ver `Entrenamiento.py`), que es un modelo lineal de 5 coeficientes ya
muy difícil de batir. La red solo tiene que aprender lo que el HAR-X no
captura, que es una tarea de varianza mucho menor y evita que una red con
miles de parámetros compita contra un baseline casi óptimo.

Dimensionamiento: con un target de RV a 5 días solapado, ~9.000 días hábiles
equivalen a solo ~1.800 observaciones independientes. Por eso la red es
deliberadamente pequeña (~2.000 parámetros); una LSTM(64) como la original
tenía 21.569 parámetros y quedaba sobre-parametrizada por un factor de ~12.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Average, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential


# Hiperparámetros de la arquitectura (se guardan en el config del modelo)
UNIDADES_LSTM = 16
UNIDADES_DENSA = 8
DROPOUT = 0.1


def construir_lstm(n_pasos, n_caracteristicas, unidades=UNIDADES_LSTM,
                   semilla=None, nombre="lstm"):
    """
    Red recurrente compacta que predice el residual estandarizado del HAR-X.

    No se usa `recurrent_dropout`: desactiva los kernels cuDNN (entrenamiento
    varias veces más lento) y en muestras pequeñas rara vez aporta sobre un
    Dropout normal.
    """
    if semilla is not None:
        tf.keras.utils.set_random_seed(int(semilla))

    modelo = Sequential(
        [
            Input(shape=(n_pasos, n_caracteristicas)),
            LSTM(unidades, activation="tanh"),
            Dropout(DROPOUT),
            Dense(UNIDADES_DENSA, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name=nombre,
    )

    # Huber sobre el residual estandarizado: menos sensible que el MSE a los
    # días de shock (COVID, 2008), que de otro modo dominan el gradiente.
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return modelo


def ensamblar(modelos, n_pasos, n_caracteristicas):
    """
    Promedia varias redes entrenadas con semillas distintas en un único modelo.

    Con este tamaño de muestra la varianza entre semillas es grande: una sola
    corrida no es un resultado, es una realización. El promedio del ensamble
    reduce esa varianza y se guarda como un solo artefacto `.keras`, así que
    la inferencia sigue siendo una única llamada a `predict`.
    """
    if len(modelos) == 1:
        return modelos[0]

    entrada = Input(shape=(n_pasos, n_caracteristicas), name="entrada")
    salidas = [m(entrada) for m in modelos]
    return Model(entrada, Average(name="promedio")(salidas), name="ensamble_lstm")


def n_parametros(modelo):
    """Número de parámetros entrenables (para reportarlo en el log)."""
    return int(np.sum([np.prod(v.shape) for v in modelo.trainable_weights]))


# ==========================================
# BLOQUE DE PRUEBA / EJECUCIÓN AISLADA
# ==========================================
if __name__ == "__main__":
    print("--- PRUEBA DE ARQUITECTURA ---")
    n_pasos, n_feats = 22, 13

    red = construir_lstm(n_pasos, n_feats, semilla=0)
    red.summary()
    print(f"\n[+] Parámetros entrenables: {n_parametros(red):,}")

    ensamble = ensamblar(
        [construir_lstm(n_pasos, n_feats, semilla=k, nombre=f"lstm_semilla_{k}")
         for k in range(3)],
        n_pasos,
        n_feats,
    )
    salida = ensamble.predict(np.random.randn(4, n_pasos, n_feats), verbose=0)
    print(f"[+] Ensamble de 3 redes -> salida {salida.shape}")
    print("[+] Arquitectura validada.")

"""
Calendario de eventos programados y variables de "evento en la ventana".

Idea: la volatilidad de la semana próxima es en parte predecible porque
algunos choques tienen FECHA CONOCIDA de antemano (decisiones de la Fed,
elecciones en EE. UU.). Las variables de rezago (RV, semivarianzas) y el VIX a
30 días no ven bien esos eventos: el VIX reparte un evento de un día entre
~21 sesiones, mientras que el objetivo del modelo es la RV de 5 días.

Reglas para evitar look-ahead (fuga de información):

1. Solo entran eventos cuya fecha se conocía con antelación. Las reuniones
   NO programadas de la Fed (conference calls, recortes de emergencia como
   marzo de 2020) se excluyen: incluirlas sería usar el choque para
   predecirse a sí mismo.
2. Solo entran CLASES completas de eventos definidas por una regla (todas las
   reuniones programadas, todas las elecciones), nunca una lista elegida a
   mano de "eventos que resultaron volátiles", que sesgaría el coeficiente.
3. La ventana de cada fila t son los 5 días hábiles siguientes, (t, t+5BD],
   calculada con días hábiles del calendario y no con el índice de datos. Así
   la última fila (la que se pronostica) usa exactamente la misma definición
   que el entrenamiento, aunque esos días todavía no existan en los datos.

Antes de febrero de 1994 la Fed no anunciaba sus decisiones (el mercado las
inferia de las operaciones de mercado abierto), así que el calendario de FOMC
arranca en 1994.

El calendario de FOMC se versiona en `calendario_eventos.csv`. Para
regenerarlo (por ejemplo, cuando la Fed publique las fechas de un año nuevo)
ejecutar `python Eventos.py`.
"""

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

_CARPETA = Path(__file__).resolve().parent
ARCHIVO_CALENDARIO = _CARPETA / "calendario_eventos.csv"

HORIZONTE_EVENTOS = 5  # debe coincidir con Caracteristicas.HORIZONTE

# Variables de evento que produce `variables_eventos`
COLUMNAS_EVENTOS = ["Ev_FOMC", "Ev_Elec_Pres", "Ev_Elec_Inter"]

_ANIO_INICIO_FOMC = 1994
_URL_HISTORICO = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{anio}.htm"
_URL_ACTUAL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

_MESES = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


# ==========================================
# 1. Elecciones (regla exacta, no requiere datos)
# ==========================================
def fechas_elecciones(anio_inicio=1990, anio_fin=2040):
    """
    Elecciones federales de EE. UU.: el martes siguiente al primer lunes de
    noviembre de cada año par. Presidenciales si el año es múltiplo de 4.

    Se devuelve el MIÉRCOLES siguiente como fecha del evento: es la primera
    sesión que cotiza el resultado. Con eso, la ventana (t, t+5BD] de las
    filas del lunes y del martes electoral contiene la reacción.
    """
    filas = []
    for anio in range(anio_inicio + anio_inicio % 2, anio_fin + 1, 2):
        primero = pd.Timestamp(anio, 11, 1)
        lunes = primero + pd.Timedelta(days=(0 - primero.weekday()) % 7)
        martes = lunes + pd.Timedelta(days=1)
        tipo = "eleccion_presidencial" if anio % 4 == 0 else "eleccion_intermedia"
        filas.append((martes + pd.Timedelta(days=1), tipo, f"Elección {martes:%Y-%m-%d}"))
    return pd.DataFrame(filas, columns=["fecha", "tipo", "descripcion"])


# ==========================================
# 2. FOMC (desde el sitio de la Fed)
# ==========================================
def _html(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def _mes(texto):
    return _MESES[texto.strip().lower()[:3]]


def _fomc_historico(anio):
    """Páginas fomchistoricalYYYY: un <h5> por reunión, p. ej.
    'January 31-February 1 Meeting - 2006'. El día de la decisión es el último."""
    fechas = []
    for m in re.finditer(r"<h5[^>]*>(.*?)</h5>", _html(_URL_HISTORICO.format(anio=anio)), re.S):
        titulo = re.sub(r"<[^>]+>|\s+", " ", m.group(1)).strip()
        if "Meeting" not in titulo or "nscheduled" in titulo:
            continue  # conference calls y reuniones de emergencia: no programadas
        partes = re.findall(r"([A-Z][a-z]+)\s+(\d{1,2})(?:-(\d{1,2}))?",
                            titulo.split("Meeting")[0])
        if not partes:
            continue
        mes, d1, d2 = partes[-1]
        fechas.append(pd.Timestamp(anio, _mes(mes), int(d2 or d1)))
    return fechas


def _fomc_actual():
    """Página fomccalendars: un panel por año con pares mes/días.
    Incluye los años futuros ya publicados por la Fed."""
    html = _html(_URL_ACTUAL)
    fechas = []
    bloques = re.split(r"(\d{4}) FOMC Meetings", html)
    for anio, cuerpo in zip(bloques[1::2], bloques[2::2]):
        anio = int(anio)
        patron = (r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?'
                  r'fomc-meeting__date[^>]*>([^<]+)<')
        for mes_txt, dias_txt in re.findall(patron, cuerpo, re.S):
            if "nscheduled" in dias_txt or "notation" in dias_txt.lower():
                continue
            dias = re.findall(r"\d{1,2}", dias_txt)
            if not dias:
                continue
            mes = mes_txt.split("/")[-1]  # "Apr/May" -> la decisión es en mayo
            fechas.append(pd.Timestamp(anio, _mes(mes), int(dias[-1])))
    return fechas


def descargar_calendario_fomc(anio_fin_historico=None):
    """Decisiones programadas del FOMC desde 1994 hasta lo publicado por la Fed."""
    actuales = _fomc_actual()
    primer_actual = min(f.year for f in actuales)
    anio_fin_historico = anio_fin_historico or primer_actual - 1
    fechas = []
    for anio in range(_ANIO_INICIO_FOMC, anio_fin_historico + 1):
        fechas.extend(_fomc_historico(anio))
    fechas.extend(f for f in actuales if f.year > anio_fin_historico)
    fechas = sorted(set(fechas))
    return pd.DataFrame({"fecha": fechas, "tipo": "fomc",
                         "descripcion": "Decisión FOMC programada"})


def guardar_calendario():
    """Regenera `calendario_eventos.csv` (FOMC). Las elecciones salen de la regla."""
    cal = descargar_calendario_fomc()
    # Sanidad: 8 reuniones programadas por año desde 1994 (algunos años, 9).
    por_anio = cal.groupby(cal["fecha"].dt.year).size()
    raros = por_anio[(por_anio < 7) | (por_anio > 9)]
    if len(raros):
        raise ValueError(f"Conteo de reuniones FOMC sospechoso: {raros.to_dict()}")
    cal.to_csv(ARCHIVO_CALENDARIO, index=False, date_format="%Y-%m-%d")
    return cal


def cargar_calendario():
    """Calendario completo: FOMC versionado + elecciones por regla."""
    partes = [fechas_elecciones()]
    if ARCHIVO_CALENDARIO.exists():
        fomc = pd.read_csv(ARCHIVO_CALENDARIO, parse_dates=["fecha"])
        partes.append(fomc)
    cal = pd.concat(partes, ignore_index=True)
    return cal.sort_values("fecha").reset_index(drop=True)


# ==========================================
# 3. Variables para el modelo
# ==========================================
_TIPO_A_COLUMNA = {
    "fomc": "Ev_FOMC",
    "eleccion_presidencial": "Ev_Elec_Pres",
    "eleccion_intermedia": "Ev_Elec_Inter",
}


def variables_eventos(indice, calendario=None, horizonte=HORIZONTE_EVENTOS):
    """
    Para cada fecha t del índice: 1 si un evento de cada tipo cae en la
    ventana (t, t + horizonte días hábiles], 0 si no.

    Si el calendario de FOMC no cubre una fecha (archivo ausente o fechas
    posteriores a lo publicado por la Fed), la variable vale 0 ahí; por eso
    conviene regenerar el CSV cada año.
    """
    indice = pd.DatetimeIndex(indice).normalize()
    cal = cargar_calendario() if calendario is None else calendario
    fin = indice + BDay(horizonte)
    salida = pd.DataFrame(0.0, index=indice, columns=COLUMNAS_EVENTOS)

    for tipo, columna in _TIPO_A_COLUMNA.items():
        fechas = np.sort(cal.loc[cal["tipo"] == tipo, "fecha"].to_numpy(dtype="datetime64[ns]"))
        if len(fechas) == 0:
            continue
        # número de eventos en (t, fin]: busca por ordenamiento, O(n log m)
        n_hasta_fin = np.searchsorted(fechas, fin.to_numpy(), side="right")
        n_hasta_t = np.searchsorted(fechas, indice.to_numpy(), side="right")
        salida[columna] = ((n_hasta_fin - n_hasta_t) > 0).astype(float)

    return salida


if __name__ == "__main__":
    cal = guardar_calendario()
    print(f"[+] {len(cal)} decisiones FOMC programadas "
          f"({cal['fecha'].min():%Y-%m-%d} → {cal['fecha'].max():%Y-%m-%d})")
    print(f"[+] Guardado en {ARCHIVO_CALENDARIO.name}")

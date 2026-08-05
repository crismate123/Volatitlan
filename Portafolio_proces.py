import pandas as pd
import yfinance as yf
from tqdm import tqdm
import requests
from io import StringIO

def get_top_80_sp500_by_market_cap():
    # 1. Obtener la lista de empresas del S&P 500 desde Wikipedia
    print("Obteniendo la lista de empresas del S&P 500...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # --- SOLUCIÓN AL ERROR 403 ---
    # Enviamos un 'User-Agent' haciéndonos pasar por un navegador web
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Hacemos la petición con requests
    response = requests.get(url, headers=headers)
    
    # Usamos StringIO para que pandas lea el texto HTML sin generar advertencias
    table = pd.read_html(StringIO(response.text))
    df = table[0]
    # -----------------------------
    
    # Extraer los símbolos y los nombres de las empresas
    companies = dict(zip(df['Symbol'], df['Security']))
    tickers = list(companies.keys())

    # Yahoo Finance usa guiones en lugar de puntos para ciertas acciones
    tickers = [ticker.replace('.', '-') for ticker in tickers]

    # 2. Obtener la capitalización de mercado para cada empresa
    data = []
    print("Descargando capitalización de mercado (esto tomará unos minutos)...")
    
    for ticker in tqdm(tickers):
        try:
            stock = yf.Ticker(ticker)
            mcap = stock.info.get('marketCap')
            
            if mcap:
                original_ticker = ticker.replace('-', '.')
                data.append({
                    'Ticker': original_ticker,
                    'Empresa': companies.get(original_ticker, ticker),
                    'Market Cap (USD)': mcap
                })
        except Exception as e:
            continue

    # 3. Crear un DataFrame con los datos recolectados
    mcap_df = pd.DataFrame(data)

    # Ordenar de mayor a menor por Market Cap y reiniciar el índice
    mcap_df = mcap_df.sort_values(by='Market Cap (USD)', ascending=False).reset_index(drop=True)

    # 4. Seleccionar las 80 empresas más grandes
    top_80 = mcap_df.head(80)

    # Formatear la columna de Market Cap
    top_80.loc[:, 'Market Cap (Billions USD)'] = (top_80['Market Cap (USD)'] / 1e9).apply(lambda x: f"${x:,.2f}B")
    top_80 = top_80.drop(columns=['Market Cap (USD)'])

    return top_80

if __name__ == "__main__":
    top_80_companies = get_top_80_sp500_by_market_cap()
    
    print("\n--- Top 80 Empresas del S&P 500 por Valor de Mercado ---")
    pd.set_option('display.max_rows', 80)
    print(top_80_companies)
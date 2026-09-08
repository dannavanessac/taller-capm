import os
import re
import pandas as pd
import numpy as np
from scipy import stats

ASSET_METADATA = {
    "BTC": {
        "folder": "BTC",
        "name": "Bitcoin (BTC-USD)",
        "category": "Criptoactivo",
        "description": "Criptomoneda líder por capitalización y liquidez global.",
        "p0": 48960.789062,
        "is_crypto": True,
        "has_volume": True
    },
    "GOSS": {
        "folder": "GOSS",
        "name": "Gossamer Bio, Inc. (GOSS)",
        "category": "Acción Biotecnológica (Stock)",
        "description": "Empresa biofarmacéutica en fase clínica orientada al desarrollo de fármacos.",
        "p0": 9.58,
        "is_crypto": False,
        "has_volume": True
    },
    "NBI": {
        "folder": "NBI",
        "name": "Nasdaq Biotechnology Index (^NBI)",
        "category": "Índice Sectorial Biotecnología",
        "description": "Índice bursátil representativo de la industria biotecnológica y farmacéutica.",
        "p0": 5343.640137,
        "is_crypto": False,
        "has_volume": True
    },
    "NCI": {
        "folder": "NCI",
        "name": "Nasdaq Crypto Index (NCI)",
        "category": "Índice Cripto",
        "description": "Índice de referencia ponderado del mercado de activos digitales.",
        "p0": 2989.34,
        "is_crypto": True,
        "has_volume": False
    },
    "S&P500": {
        "folder": "S&P500",
        "name": "S&P 500 Index (^GSPC)",
        "category": "Índice de Mercado General",
        "description": "Principal índice de referencia accionario de Estados Unidos (500 mayores empresas).",
        "p0": 4496.189941,
        "is_crypto": False,
        "has_volume": True
    },
    "XBI": {
        "folder": "XBI",
        "name": "SPDR S&P Biotech ETF (XBI)",
        "category": "ETF Sectorial Biotecnología",
        "description": "ETF diversificado de igual ponderación del sector biotecnológico.",
        "p0": 128.344681,
        "is_crypto": False,
        "has_volume": True
    }
}

def clean_stats_value(v_str):
    """Limpia cadenas de pandas Series dentro de *_stats.csv."""
    if pd.isna(v_str):
        return np.nan
    s = str(v_str).strip()
    if "Ticker" in s or "dtype:" in s:
        match = re.search(r"([-\d\.]+)\s*\n\s*dtype:", s)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
        if numbers:
            return float(numbers[-1])
    try:
        return float(s)
    except:
        return s

def load_extracted_stats(base_dir="."):
    """Extrae las tablas de estadísticas originales de cada subcarpeta."""
    all_stats = {}
    for ticker, info in ASSET_METADATA.items():
        folder = os.path.join(base_dir, info["folder"])
        csv_candidates = [
            os.path.join(folder, f"{ticker.lower()}_stats.csv"),
            os.path.join(folder, "gspc_stats.csv" if ticker == "S&P500" else f"{ticker.lower()}_stats.csv")
        ]
        stats_path = None
        for cand in csv_candidates:
            if os.path.exists(cand):
                stats_path = cand
                break
        
        if stats_path and os.path.exists(stats_path):
            try:
                raw_df = pd.read_csv(stats_path, encoding='latin-1')
            except:
                raw_df = pd.read_csv(stats_path, encoding='utf-8')
            
            raw_df.columns = ["Métrica", "Valor Original"]
            raw_df["Valor Numérico"] = raw_df["Valor Original"].apply(clean_stats_value)
            all_stats[ticker] = raw_df
        else:
            all_stats[ticker] = pd.DataFrame()
            
    return all_stats

def load_all_series(base_dir="."):
    """
    Carga y estandariza las series de tiempo de cada activo.
    Resuelve inconsistencias de formato de fecha, ordenamiento y normalización de precios.
    """
    data_dict = {}
    for ticker, info in ASSET_METADATA.items():
        folder = os.path.join(base_dir, info["folder"])
        file_name = "df_gspc.csv" if ticker == "S&P500" else f"df_{ticker.lower()}.csv"
        path = os.path.join(folder, file_name)
        
        if not os.path.exists(path):
            continue
            
        df = pd.read_csv(path, sep=';', encoding='utf-8')
        
        # 1. Parsear Fechas uniformemente
        df['Date_Parsed'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        if df['Date_Parsed'].isna().sum() > len(df) * 0.5:
            df['Date_Parsed'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        if df['Date_Parsed'].isna().sum() > 0:
            df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
            
        # Ordenar cronológicamente ascendente
        df = df.sort_values('Date_Parsed').reset_index(drop=True)
        
        # 2. Corregir y limpiar retornos logarítmicos
        if ticker == "GOSS":
            for idx in df.index:
                val = str(df.at[idx, 'Log_Returns'])
                if "-13.702" in val:
                    df.at[idx, 'Log_Returns'] = -1.37027697413291
                elif "-1.616" in val and len(val) > 10:
                    df.at[idx, 'Log_Returns'] = -1.61650511478529
        
        if ticker == "NCI":
            df['Close_Clean'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Log_Returns_Clean'] = np.log(df['Close_Clean'] / df['Close_Clean'].shift(1))
            for col in ['Open', 'High', 'Low']:
                if col in df.columns:
                    df[f'{col}_Clean'] = pd.to_numeric(df[col], errors='coerce')
        else:
            df['Log_Returns_Clean'] = pd.to_numeric(df['Log_Returns'], errors='coerce')
            df['Close_Clean'] = info['p0'] * np.exp(df['Log_Returns_Clean'].fillna(0).cumsum())
            
        # 3. Retornos Simples y Retorno Acumulado
        df['Simple_Returns'] = np.exp(df['Log_Returns_Clean']) - 1.0
        first_close = df['Close_Clean'].iloc[0] if len(df) > 0 and pd.notna(df['Close_Clean'].iloc[0]) and df['Close_Clean'].iloc[0] != 0 else 1.0
        df['Cumulative_Return'] = (df['Close_Clean'] / first_close) - 1.0
        df['Normalized_100'] = (df['Close_Clean'] / first_close) * 100.0
        
        # 4. Medias Móviles de Precio
        df['SMA_20'] = df['Close_Clean'].rolling(window=20).mean()
        df['SMA_50'] = df['Close_Clean'].rolling(window=50).mean()
        df['SMA_200'] = df['Close_Clean'].rolling(window=200).mean()
        
        # 5. Volatilidad Móvil Anualizada (30 y 90 días)
        ann_factor = np.sqrt(365 if info['is_crypto'] else 252)
        df['Rolling_Vol_30d'] = df['Log_Returns_Clean'].rolling(window=30).std() * ann_factor
        df['Rolling_Vol_90d'] = df['Log_Returns_Clean'].rolling(window=90).std() * ann_factor
        
        # 6. Drawdowns (Caídas desde el pico histórico anterior)
        df['Peak_Price'] = df['Close_Clean'].cummax()
        df['Drawdown'] = (df['Close_Clean'] - df['Peak_Price']) / df['Peak_Price']
        
        # 7. Volumen de Negociación Limpio y Media Móvil de Volumen
        if 'Volume' in df.columns:
            # Reemplazar comas decimales o notaciones científicas con coma
            vol_str = df['Volume'].astype(str).str.replace(',', '.')
            df['Volume_Clean'] = pd.to_numeric(vol_str, errors='coerce')
            df['Volume_SMA20'] = df['Volume_Clean'].rolling(window=20).mean()
        else:
            df['Volume_Clean'] = np.nan
            df['Volume_SMA20'] = np.nan
            
        data_dict[ticker] = df
        
    return data_dict

def calculate_kpis(df, is_crypto=False, risk_free_rate=0.04):
    """
    Calcula KPIs cuantitativos sobre una serie temporal para un rango dado.
    """
    if df.empty or len(df) < 5:
        return {}
        
    returns = df['Log_Returns_Clean'].dropna()
    ann_days = 365 if is_crypto else 252
    
    # Precios y Retornos
    p_start = df['Close_Clean'].iloc[0]
    p_end = df['Close_Clean'].iloc[-1]
    p_max = df['Close_Clean'].max()
    p_min = df['Close_Clean'].min()
    total_cum_ret = (p_end / p_start) - 1.0
    
    mean_daily = returns.mean()
    ann_ret = mean_daily * ann_days
    
    # Volatilidad y Dispersión
    std_daily = returns.std()
    ann_vol = std_daily * np.sqrt(ann_days)
    variance_daily = returns.var()
    skewness = returns.skew()
    kurtosis = returns.kurt()
    
    # Ratios de Eficiencia
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    
    # Drawdowns
    drawdown_series = df['Drawdown']
    max_drawdown = drawdown_series.min()
    
    # Value at Risk (Histórico al 95% y 99% a 1 día)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Expected Shortfall (CVaR 95%)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    
    # Calmar Ratio
    calmar = (ann_ret / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0
    
    # Promedio de volumen
    vol_mean = df['Volume_Clean'].mean() if 'Volume_Clean' in df.columns else np.nan
    
    return {
        "p_start": p_start,
        "p_end": p_end,
        "p_max": p_max,
        "p_min": p_min,
        "total_cum_ret": total_cum_ret,
        "mean_daily": mean_daily,
        "ann_ret": ann_ret,
        "std_daily": std_daily,
        "ann_vol": ann_vol,
        "variance_daily": variance_daily,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "calmar": calmar,
        "vol_mean": vol_mean,
        "count": len(returns),
        "date_start": df['Date_Parsed'].min(),
        "date_end": df['Date_Parsed'].max()
    }

def generate_financial_diagnosis(ticker, kpis):
    """
    Genera un diagnóstico narrativo y conclusiones del comportamiento intrínseco del activo,
    enfocado exclusivamente en rentabilidad, riesgo, volatilidad, distribución y caídas.
    """
    info = ASSET_METADATA.get(ticker, {})
    name = info.get("name", ticker)
    
    conclusions = []
    
    # 1. Desempeño y Tendencia de Precio
    cum_ret = kpis["total_cum_ret"]
    ann_ret = kpis["ann_ret"]
    p_start = kpis["p_start"]
    p_end = kpis["p_end"]
    p_max = kpis["p_max"]
    p_min = kpis["p_min"]
    
    if cum_ret > 0.5:
        conclusions.append(f"**Crecimiento Extraordinario de Valor**: {name} acumuló una rentabilidad de **{cum_ret*100:+.2f}%** ({ann_ret*100:+.2f}% anualizado), pasando de ${p_start:,.2f} a ${p_end:,.2f}, con un máximo histórico en el período de ${p_max:,.2f}.")
    elif cum_ret > 0:
        conclusions.append(f"**Rendimiento Positivo**: El activo registró una rentabilidad acumulada de **{cum_ret*100:+.2f}%** ({ann_ret*100:+.2f}% anualizado), oscilando entre un mínimo de ${p_min:,.2f} y un máximo de ${p_max:,.2f}.")
    else:
        conclusions.append(f"**Contracción Severa de Valor**: El activo experimentó una pérdida acumulada del **{cum_ret*100:+.2f}%** ({ann_ret*100:+.2f}% anualizado), cayendo desde un precio inicial de ${p_start:,.2f} hasta ${p_end:,.2f} (mínimo registrado de ${p_min:,.2f}).")

    # 2. Volatilidad y Régimen de Riesgo
    ann_vol = kpis["ann_vol"]
    std_d = kpis["std_daily"]
    if ann_vol > 0.6:
        conclusions.append(f"**Régimen de Hiper-Volatilidad**: Con una volatilidad anualizada de **{ann_vol*100:.2f}%** (desviación diaria de {std_d*100:.2f}%), el activo presenta fluctuaciones de precio muy pronunciadas que implican un alto grado de incertidumbre.")
    elif ann_vol > 0.25:
        conclusions.append(f"**Volatilidad Dinámica Moderada-Alta**: Su volatilidad anualizada es de **{ann_vol*100:.2f}%** (desviación diaria de {std_d*100:.2f}%), comportamiento habitual en activos de renta variable y sectores innovadores.")
    else:
        conclusions.append(f"**Volatilidad Moderada y Estable**: Registra una volatilidad anualizada contenida de **{ann_vol*100:.2f}%** ({std_d*100:.2f}% diario), ofreciendo una trayectoria de precios considerablemente más suave y predecible.")

    # 3. Asimetría, Curtosis y Riesgo de Cola (Fat Tails)
    skew = kpis["skewness"]
    kurt = kpis["kurtosis"]
    if kurt > 5.0 or abs(skew) > 1.0:
        skew_type = "sesgo fuertemente negativo (caídas abruptas más frecuentes que subidas equivalentes)" if skew < -0.5 else ("sesgo positivo (disparos alcistas atípicos)" if skew > 0.5 else "distribución relativamente simétrica")
        conclusions.append(f"**Comportamiento No Normal (Colas Pesadas)**: Presenta una curtosis de **{kurt:.2f}** (muy superior al valor 3 de una curva normal) y una asimetría de **{skew:.2f}** ({skew_type}). Esto comprueba la presencia recurrente de movimientos atípicos extremos ('cisnes negros'), por lo que los modelos estadísticos gaussianos subestiman su riesgo real en eventos de crisis.")
    else:
        conclusions.append(f"**Distribución de Retornos Cercana a la Normalidad**: Exhibe una curtosis de **{kurt:.2f}** y asimetría de **{skew:.2f}**, lo que refleja una frecuencia moderada de choques atípicos.")

    # 4. Caída Máxima y Preservación de Capital (Drawdown)
    mdd = kpis["max_drawdown"]
    calmar = kpis["calmar"]
    if abs(mdd) > 0.6:
        conclusions.append(f"**Exposición Crítica a Pérdidas (Drawdown Severo)**: La caída máxima desde su punto más alto fue de **{mdd*100:.2f}%**. Una caída de esta magnitud demanda una recuperación de más del {((1/(1+mdd))-1)*100:.0f}% solo para volver al nivel previo.")
    elif abs(mdd) > 0.3:
        conclusions.append(f"**Correcciones Cíclicas Significativas**: Sufrió una caída máxima de **{mdd*100:.2f}%** durante fases correctivas, con un Ratio de Calmar de **{calmar:.2f}**.")
    else:
        conclusions.append(f"**Control Eficaz de Caídas**: Su Máximo Drawdown se limitó al **{mdd*100:.2f}%**, demostrando alta capacidad de contención de pérdidas de capital.")

    # 5. Eficiencia Riesgo-Retorno (Sharpe Ratio)
    sharpe = kpis["sharpe"]
    if sharpe > 0.8:
        conclusions.append(f"**Excelente Rentabilidad Ajustada por Riesgo**: Generó un Sharpe Ratio de **{sharpe:.2f}**, evidenciando que el rendimiento obtenido compensó con holgura la volatilidad soportada.")
    elif sharpe > 0:
        conclusions.append(f"**Rendimiento Ajustado por Riesgo Positivo**: Sharpe Ratio de **{sharpe:.2f}**, superando el costo de oportunidad del activo libre de riesgo.")
    else:
        conclusions.append(f"**Rendimiento Ajustado por Riesgo Deficitario**: Sharpe Ratio negativo de **{sharpe:.2f}**, indicando que la rentabilidad no alcanzó para cubrir la tasa libre de riesgo ni compensar la volatilidad del activo.")

    # 6. Dinámica de Liquidez
    if not np.isnan(kpis["vol_mean"]) and kpis["vol_mean"] > 0:
        conclusions.append(f"**Actividad Transaccional**: Promedio de volumen diario negociado de **{kpis['vol_mean']:,.0f}** unidades, garantizando liquidez continua en el mercado.")

    return conclusions

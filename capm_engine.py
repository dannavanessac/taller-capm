import os
import urllib.request
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# Metadatos de modelos CAPM
CAPM_SPECS = {
    "GOSS": {
        "name": "Gossamer Bio, Inc. (GOSS)",
        "asset_file": "GOSS/df_goss.csv",
        "date_format": "%d/%m/%Y",
        "return_col": "Log_Returns",
        "benchmarks": {
            "NBI": {"file": "NBI/df_nbi.csv", "name": "Nasdaq Biotechnology Index (^NBI)", "date_format": "%d/%m/%Y"},
            "XBI": {"file": "XBI/df_xbi.csv", "name": "SPDR S&P Biotech ETF (XBI)", "date_format": "%d/%m/%Y"},
            "S&P 500": {"file": "S&P500/df_gspc.csv", "name": "S&P 500 Index (^GSPC)", "date_format": "%d/%m/%Y"}
        }
    },
    "BTC": {
        "name": "Bitcoin (BTC-USD)",
        "asset_file": "BTC/df_btc.csv",
        "date_format": "%d/%m/%Y",
        "return_col": "Log_Returns",
        "benchmarks": {
            "NCI": {"file": "NCI/df_nci.csv", "name": "Nasdaq Crypto Index (NCI)", "date_format": "%m/%d/%Y"},
            "S&P 500": {"file": "S&P500/df_gspc.csv", "name": "S&P 500 Index (^GSPC)", "date_format": "%d/%m/%Y"}
        }
    }
}

def load_rf_data(csv_path="dtb3.csv", start_date='2021-08-25', end_date='2026-08-24'):
    """
    Carga la tasa libre de riesgo T-Bill a 3 meses (DTB3) de FRED.
    Usa el archivo local dtb3.csv o lo descarga de FRED si no existe.
    """
    if os.path.exists(csv_path):
        df_rf = pd.read_csv(csv_path)
    else:
        url_fred = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd={start_date}&coed={end_date}"
        df_rf = pd.read_csv(url_fred)
        df_rf.to_csv(csv_path, index=False)
        
    date_col = "observation_date" if "observation_date" in df_rf.columns else "Date"
    val_col = "DTB3" if "DTB3" in df_rf.columns else "rf_percent"
    
    df_rf["Date"] = pd.to_datetime(df_rf[date_col], errors="coerce")
    df_rf["rf_percent"] = pd.to_numeric(df_rf[val_col], errors="coerce")
    df_rf = df_rf.dropna(subset=["Date", "rf_percent"]).copy()
    
    # 1. Porcentaje a decimal anual
    df_rf["rf_annual"] = df_rf["rf_percent"] / 100.0
    # 2. Tasa diaria usando convención de 252 días hábiles (como en los notebooks)
    df_rf["rf_daily"] = df_rf["rf_annual"] / 252.0
    # Tasa efectiva diaria alternativa
    df_rf["rf_daily_eff"] = (1.0 + df_rf["rf_annual"]) ** (1.0 / 252.0) - 1.0
    
    df_rf = df_rf.sort_values("Date").reset_index(drop=True)
    return df_rf[["Date", "rf_percent", "rf_annual", "rf_daily", "rf_daily_eff"]]

def prepare_returns_series(file_path, ticker, date_format="%d/%m/%Y"):
    """
    Carga y estandariza los retornos de un activo o benchmark.
    Corrige posibles artefactos numéricos de Excel.
    """
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    df['Date'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
    if df['Date'].isna().sum() > len(df) * 0.5:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # Limpieza de Log_Returns
    if 'Log_Returns' in df.columns:
        s_lr = df['Log_Returns'].astype(str)
        # Corrección de puntos de miles en GOSS si existen
        if ticker == "GOSS":
            for idx in df.index:
                val = s_lr.loc[idx]
                if "-13.702" in val:
                    df.at[idx, 'Log_Returns'] = -1.37027697413291
                elif "-1.616" in val and len(val) > 10:
                    df.at[idx, 'Log_Returns'] = -1.61650511478529
        df['Log_Returns'] = pd.to_numeric(df['Log_Returns'], errors='coerce')
    else:
        df['Log_Returns'] = np.nan
        
    # En caso de NCI, si faltaran retornos o viniera invertido
    if ticker == "NCI" and ('Close' in df.columns or 'Close/Last' in df.columns):
        col_c = 'Close' if 'Close' in df.columns else 'Close/Last'
        df['Close_num'] = pd.to_numeric(df[col_c], errors='coerce')
        df['Log_Returns'] = np.log(df['Close_num'] / df['Close_num'].shift(1))
        
    df = df.rename(columns={'Log_Returns': f'r_{ticker}'})
    return df[['Date', f'r_{ticker}']].drop_duplicates('Date').dropna()

def build_aligned_capm_dataset(asset_ticker, market_ticker, df_rf, start_date=None, end_date=None):
    """
    Construye la base común alineada (Inner Join) entre activo, mercado y T-Bill.
    Sin imputación de fechas faltantes, reproduciendo con exactitud los notebooks.
    """
    asset_spec = CAPM_SPECS[asset_ticker]
    market_spec = asset_spec["benchmarks"][market_ticker]
    
    asset_df = prepare_returns_series(asset_spec["asset_file"], asset_ticker, asset_spec["date_format"])
    market_df = prepare_returns_series(market_spec["file"], market_ticker if market_ticker != "S&P 500" else "SP500", market_spec["date_format"])
    
    mkt_key = market_ticker if market_ticker != "S&P 500" else "SP500"
    
    merged = (
        asset_df
        .merge(market_df, on='Date', how='inner')
        .merge(df_rf[['Date', 'rf_daily']], on='Date', how='inner')
        .sort_values('Date')
        .reset_index(drop=True)
    )
    
    # Filtrar por rango de fechas si se especifica
    if start_date is not None:
        merged = merged[merged['Date'].dt.date >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        merged = merged[merged['Date'].dt.date <= pd.to_datetime(end_date).date()]
        
    merged = merged.dropna(subset=[f'r_{asset_ticker}', f'r_{mkt_key}', 'rf_daily']).copy()
    
    # Retornos en exceso
    merged['excess_asset'] = merged[f'r_{asset_ticker}'] - merged['rf_daily']
    merged['excess_market'] = merged[f'r_{mkt_key}'] - merged['rf_daily']
    
    return merged

def estimate_capm_model(data, hac=False, maxlags=5):
    """
    Estima la regresión lineal del CAPM por MCO:
    Y = alpha + beta * X + epsilon
    
    Permite estimar:
    - OLS Clásico (hac=False)
    - OLS con matriz Newey-West HAC (hac=True, maxlags=maxlags)
    """
    X = sm.add_constant(data['excess_market'])
    y = data['excess_asset']
    
    if hac:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    else:
        model = sm.OLS(y, X).fit()
        
    return model

def compute_downside_beta(data):
    """
    Calcula el Downside Beta (Beta Bajista) cuando el exceso de retorno del mercado es negativo:
    Cov(R_i - Rf, R_m - Rf | R_m - Rf < 0) / Var(R_m - Rf | R_m - Rf < 0)
    """
    down = data[data['excess_market'] < 0]
    if len(down) < 3 or down['excess_market'].var(ddof=1) == 0:
        return np.nan
    return down['excess_asset'].cov(down['excess_market']) / down['excess_market'].var(ddof=1)

def compute_aligned_risk_metrics(data, asset_ticker):
    """
    Calcula las métricas integrales de riesgo sobre la misma muestra alineada del modelo:
    Sharpe Ratio, Sortino Ratio, Máximo Drawdown, VaR 95% y CVaR 95%.
    """
    TRADING_DAYS = 252
    excess = data['excess_asset']
    r = data[f'r_{asset_ticker}']
    
    sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(TRADING_DAYS) if excess.std(ddof=1) > 0 else np.nan
    
    # Downside deviation para Sortino
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    sortino = (excess.mean() / downside_dev) * np.sqrt(TRADING_DAYS) if downside_dev > 0 else np.nan
    
    # Riqueza y Drawdown
    wealth = np.exp(r.cumsum())
    drawdown = wealth / wealth.cummax() - 1.0
    max_dd = drawdown.min()
    
    # VaR 95% y CVaR 95% (en retornos simples / log)
    q05 = r.quantile(0.05)
    var95 = -q05
    cvar95 = -r[r <= q05].mean()
    
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "var95": var95,
        "cvar95": cvar95,
        "wealth": wealth,
        "drawdown": drawdown
    }

def compute_residual_diagnostics(model):
    """
    Calcula diagnósticos econométricos sobre los residuos del modelo:
    - Durbin-Watson (autocorrelación de residuos)
    - Jarque-Bera (asimetría, curtosis y p-valor de normalidad)
    """
    resid = model.resid
    dw = durbin_watson(resid)
    jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid)
    return {
        "durbin_watson": dw,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "resid_skew": skew,
        "resid_kurt": kurt
    }

def run_comprehensive_capm_analysis(hac_lags=5, start_date=None, end_date=None):
    """
    Ejecuta todos los modelos de GOSS y BTC para ambas especificaciones (Sin HAC y Con HAC),
    reproduciendo las tablas finales integrales y comparativas de los cuadernos.
    """
    df_rf = load_rf_data(start_date=start_date or '2021-08-25', end_date=end_date or '2026-08-24')
    
    all_results = []
    comparison_records = []
    models_dict = {}
    datasets_dict = {}
    
    for asset_ticker, spec in CAPM_SPECS.items():
        models_dict[asset_ticker] = {}
        datasets_dict[asset_ticker] = {}
        
        for market_ticker in spec["benchmarks"].keys():
            data = build_aligned_capm_dataset(asset_ticker, market_ticker, df_rf, start_date, end_date)
            datasets_dict[asset_ticker][market_ticker] = data
            
            # Estimación OLS Clásico
            model_ols = estimate_capm_model(data, hac=False)
            # Estimación OLS con HAC Newey-West
            model_hac = estimate_capm_model(data, hac=True, maxlags=hac_lags)
            
            models_dict[asset_ticker][market_ticker] = {
                "OLS": model_ols,
                "HAC": model_hac,
                "data": data
            }
            
            # Métricas de riesgo y Downside Beta
            risk_metrics = compute_aligned_risk_metrics(data, asset_ticker)
            beta_down = compute_downside_beta(data)
            
            # Diagnósticos de residuos
            diag_ols = compute_residual_diagnostics(model_ols)
            diag_hac = compute_residual_diagnostics(model_hac)
            
            # 1. Registros para Tabla Final Integral (con y sin HAC)
            for spec_name, mod, diag in [("Sin HAC", model_ols, diag_ols), ("HAC", model_hac, diag_hac)]:
                all_results.append({
                    "Activo": asset_ticker,
                    "Mercado": market_ticker,
                    "Modelo": spec_name,
                    "Alfa (diario)": mod.params["const"],
                    "Alfa (anual)": mod.params["const"] * 252.0,
                    "Beta": mod.params["excess_market"],
                    "Beta bajista": beta_down,
                    "R²": mod.rsquared,
                    "R² ajustado": mod.rsquared_adj,
                    "EE Alfa": mod.bse["const"],
                    "EE Beta": mod.bse["excess_market"],
                    "t/z Alfa": mod.tvalues["const"],
                    "t/z Beta": mod.tvalues["excess_market"],
                    "p-value Alfa": mod.pvalues["const"],
                    "p-value Beta": mod.pvalues["excess_market"],
                    "CI 95% Alfa Inf": mod.conf_int().loc["const", 0],
                    "CI 95% Alfa Sup": mod.conf_int().loc["const", 1],
                    "CI 95% Beta Inf": mod.conf_int().loc["excess_market", 0],
                    "CI 95% Beta Sup": mod.conf_int().loc["excess_market", 1],
                    "Sharpe": risk_metrics["sharpe"],
                    "Sortino": risk_metrics["sortino"],
                    "Máximo Drawdown": risk_metrics["max_dd"],
                    "VaR 95%": risk_metrics["var95"],
                    "CVaR 95%": risk_metrics["cvar95"],
                    "Durbin-Watson": diag["durbin_watson"],
                    "Jarque-Bera": diag["jarque_bera_stat"],
                    "p-value JB": diag["jarque_bera_pvalue"],
                    "N": int(mod.nobs)
                })
                
            # 2. Registros comparativos directos (Lado a Lado: Sin HAC vs Con HAC)
            se_alpha_ols = model_ols.bse["const"]
            se_alpha_hac = model_hac.bse["const"]
            delta_se_alpha = ((se_alpha_hac / se_alpha_ols) - 1.0) * 100.0
            
            se_beta_ols = model_ols.bse["excess_market"]
            se_beta_hac = model_hac.bse["excess_market"]
            delta_se_beta = ((se_beta_hac / se_beta_ols) - 1.0) * 100.0
            
            comparison_records.append({
                "Activo": asset_ticker,
                "Mercado": market_ticker,
                "Alfa (α)": model_ols.params["const"],
                "EE Alfa (Sin HAC)": se_alpha_ols,
                "EE Alfa (Con HAC)": se_alpha_hac,
                "Δ% EE Alfa": delta_se_alpha,
                "p-val Alfa (Sin HAC)": model_ols.pvalues["const"],
                "p-val Alfa (Con HAC)": model_hac.pvalues["const"],
                "Beta (β)": model_ols.params["excess_market"],
                "Beta Bajista": beta_down,
                "EE Beta (Sin HAC)": se_beta_ols,
                "EE Beta (Con HAC)": se_beta_hac,
                "Δ% EE Beta": delta_se_beta,
                "p-val Beta (Sin HAC)": model_ols.pvalues["excess_market"],
                "p-val Beta (Con HAC)": model_hac.pvalues["excess_market"],
                "R²": model_ols.rsquared,
                "Durbin-Watson": diag_ols["durbin_watson"],
                "Jarque-Bera": diag_ols["jarque_bera_stat"],
                "N": int(model_ols.nobs)
            })
            
    df_results = pd.DataFrame(all_results)
    df_comparison = pd.DataFrame(comparison_records)
    
    return {
        "results_df": df_results,
        "comparison_df": df_comparison,
        "models_dict": models_dict,
        "datasets_dict": datasets_dict,
        "df_rf": df_rf
    }

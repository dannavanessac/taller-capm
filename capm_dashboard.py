import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from capm_engine import (
    CAPM_SPECS,
    load_rf_data,
    run_comprehensive_capm_analysis
)

# -----------------------------------------------------------------------------
# Configuración de Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tablero de Modelación Econométrica CAPM | OLS vs HAC",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS profesionales para presentación académica/financiera
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #1E3A8A;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 500;
        margin-top: 4px;
    }
    .analysis-box {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        font-size: 1.0rem;
        line-height: 1.6;
        color: #1E293B;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 650;
        margin-right: 8px;
    }
    .badge-goss { background-color: #FEE2E2; color: #991B1B; }
    .badge-btc { background-color: #FEF3C7; color: #92400E; }
    .badge-hac { background-color: #DBEAFE; color: #1E40AF; }
    .chart-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Barra Lateral con Filtros Interactivos
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=65)
st.sidebar.title("Modelación CAPM")

# 1. Filtro de Activo
asset_choice = st.sidebar.selectbox(
    "Seleccione el Activo Analizado:",
    options=["GOSS (Gossamer Bio)", "BTC (Bitcoin)", "🌐 Vista Consolidada (GOSS y BTC)"],
    index=0,
    help="Elija qué cuaderno examinar (CAPM_GOSS o CAPM_BTC) o compare ambos activos."
)

is_consolidated = (asset_choice == "🌐 Vista Consolidada (GOSS y BTC)")
active_ticker = "GOSS" if "GOSS" in asset_choice else ("BTC" if "BTC" in asset_choice else None)

# 2. Filtro de Benchmark
if not is_consolidated:
    bench_options = ["Todos los Benchmarks"] + list(CAPM_SPECS[active_ticker]["benchmarks"].keys())
    selected_bench = st.sidebar.selectbox(
        "Mercado de Referencia (Benchmark):",
        options=bench_options,
        index=0,
        help="Filtre los modelos por índice de referencia específico."
    )
else:
    selected_bench = "Todos los Benchmarks"

# 3. Filtro de Especificación del Modelo (CON HAC vs SIN HAC)
model_spec_choice = st.sidebar.selectbox(
    "Especificación Econométrica:",
    options=["⚖️ Comparativa CON HAC vs SIN HAC", "Solo CON HAC (Newey-West)", "Solo SIN HAC (OLS Clásico)"],
    index=0,
    help="Compare directamente cómo cambian los errores estándar e inferencia entre OLS y HAC."
)

# 4. Parámetro de Rezagos Newey-West (HAC Lags)
hac_lags = st.sidebar.slider(
    "Rezagos Newey-West HAC (Lags):",
    min_value=1,
    max_value=15,
    value=5,
    help="Número de rezagos utilizados en el kernel Bartlett para estimar la matriz HAC (en los notebooks se utilizó 5)."
)

# 5. Rango Temporal
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Rango Temporal Común")

df_rf_info = load_rf_data()
min_d = df_rf_info["Date"].min().date()
max_d = df_rf_info["Date"].max().date()

date_range = st.sidebar.date_input(
    "Filtrar Período:",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_d, max_d

st.sidebar.markdown("---")
st.sidebar.caption("📘 **Taller CAPM**: Extracción directa de `CAPM_GOSS.ipynb` y `CAPM_BTC.ipynb`. Contrasta OLS Homocedástico vs Newey-West HAC y Downside Beta.")

# -----------------------------------------------------------------------------
# Ejecución del Motor Econométrico
# -----------------------------------------------------------------------------
analysis_data = run_comprehensive_capm_analysis(
    hac_lags=hac_lags,
    start_date=start_d,
    end_date=end_d
)

results_df = analysis_data["results_df"]
comparison_df = analysis_data["comparison_df"]
models_dict = analysis_data["models_dict"]
datasets_dict = analysis_data["datasets_dict"]

# Filtrar según selecciones
if not is_consolidated:
    results_df = results_df[results_df["Activo"] == active_ticker]
    comparison_df = comparison_df[comparison_df["Activo"] == active_ticker]
    if selected_bench != "Todos los Benchmarks":
        results_df = results_df[results_df["Mercado"] == selected_bench]
        comparison_df = comparison_df[comparison_df["Mercado"] == selected_bench]

if "Solo CON HAC" in model_spec_choice:
    results_df = results_df[results_df["Modelo"] == "HAC"]
elif "Solo SIN HAC" in model_spec_choice:
    results_df = results_df[results_df["Modelo"] == "Sin HAC"]

# -----------------------------------------------------------------------------
# ENCABEZADO Y TARJETAS KPI RESUMEN
# -----------------------------------------------------------------------------
title_text = "Modelación Econométrica CAPM: GOSS vs BTC" if is_consolidated else f"Modelación CAPM de {CAPM_SPECS[active_ticker]['name']}"
badge_tag = "badge-goss" if active_ticker == "GOSS" else ("badge-btc" if active_ticker == "BTC" else "badge-hac")

st.markdown(f'<div class="main-header">{title_text}</div>', unsafe_allow_html=True)
st.markdown(f'<span class="badge {badge_tag}">Rezagos HAC: {hac_lags} (Newey-West)</span> <span style="color:#64748B; font-size:1.02rem;">T-Bill 3M (FRED DTB3) como Tasa Libre de Riesgo diaria</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 4 Tarjetas Métricas Rápidas
c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)

with c_kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Modelos Estimados</div>
        <div class="kpi-value">{len(results_df)}</div>
        <div class="kpi-sub">Especificaciones activas</div>
    </div>
    """, unsafe_allow_html=True)

with c_kpi2:
    mean_beta = results_df["Beta"].mean() if not results_df.empty else np.nan
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Beta Promedio</div>
        <div class="kpi-value">{mean_beta:.3f}</div>
        <div class="kpi-sub">Sensibilidad de mercado</div>
    </div>
    """, unsafe_allow_html=True)

with c_kpi3:
    mean_beta_down = results_df["Beta bajista"].mean() if not results_df.empty else np.nan
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Beta Bajista Promedio</div>
        <div class="kpi-value" style="color:#DC2626;">{mean_beta_down:.3f}</div>
        <div class="kpi-sub">Sensibilidad con Rm < 0</div>
    </div>
    """, unsafe_allow_html=True)

with c_kpi4:
    mean_r2 = results_df["R²"].mean() if not results_df.empty else np.nan
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">R² Promedio</div>
        <div class="kpi-value" style="color:#2563EB;">{mean_r2*100:.2f}%</div>
        <div class="kpi-sub">Varianza explicada</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES DEL TABLERO
# -----------------------------------------------------------------------------
tab_comparison, tab_graphs, tab_tables, tab_theory = st.tabs([
    "⚖️ Comparación CON HAC vs SIN HAC",
    "📊 Visualizaciones y Gráficas de Modelado",
    "📋 Tablas Finales Integrales (Notebooks)",
    "💡 Interpretación y Sustentación Académica"
])

# -----------------------------------------------------------------------------
# PESTAÑA 1: COMPARACIÓN CON HAC vs SIN HAC (INFERENCIA ECONOMÉTRICA)
# -----------------------------------------------------------------------------
with tab_comparison:
    st.markdown("### ⚖️ Contraste de Inferencia Econométrica: OLS Clásico vs Newey-West HAC")
    st.caption("Esta sección permite evaluar la hipótesis central de los cuadernos: **HAC no cambia los coeficientes estimados (α, β ni R²)**, pero corrige los errores estándar y los p-valores ante la presencia de heterocedasticidad o autocorrelación serial.")
    
    # 1. Tabla Comparativa Directa
    st.markdown("#### 1. Tabla de Comparación de Errores Estándar y Significancia")
    
    display_comp = comparison_df.copy()
    format_dict = {
        "Alfa (α)": "{:.6f}",
        "EE Alfa (Sin HAC)": "{:.6f}",
        "EE Alfa (Con HAC)": "{:.6f}",
        "Δ% EE Alfa": "{:+.2f}%",
        "p-val Alfa (Sin HAC)": "{:.4f}",
        "p-val Alfa (Con HAC)": "{:.4f}",
        "Beta (β)": "{:.4f}",
        "Beta Bajista": "{:.4f}",
        "EE Beta (Sin HAC)": "{:.4f}",
        "EE Beta (Con HAC)": "{:.4f}",
        "Δ% EE Beta": "{:+.2f}%",
        "p-val Beta (Sin HAC)": "{:.4f}",
        "p-val Beta (Con HAC)": "{:.4f}",
        "R²": "{:.4f}",
        "Durbin-Watson": "{:.3f}",
        "Jarque-Bera": "{:,.1f}",
        "N": "{:.0f}"
    }
    st.dataframe(display_comp.style.format(format_dict), use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Gráfica Comparativa de Errores Estándar (Barras agrupadas)
    c_err1, c_err2 = st.columns(2)
    
    with c_err1:
        st.markdown("#### 2. Comparativa de Error Estándar de Beta: OLS vs HAC")
        fig_se_beta = go.Figure()
        
        fig_se_beta.add_trace(go.Bar(
            x=[f"{r['Activo']} vs {r['Mercado']}" for _, r in comparison_df.iterrows()],
            y=comparison_df["EE Beta (Sin HAC)"],
            name="Error Estándar (Sin HAC / OLS)",
            marker_color="#94A3B8"
        ))
        fig_se_beta.add_trace(go.Bar(
            x=[f"{r['Activo']} vs {r['Mercado']}" for _, r in comparison_df.iterrows()],
            y=comparison_df["EE Beta (Con HAC)"],
            name="Error Estándar (Con HAC / Newey-West)",
            marker_color="#2563EB"
        ))
        
        fig_se_beta.update_layout(
            barmode="group",
            yaxis_title="Error Estándar de Beta (SE)",
            template="plotly_white",
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_se_beta, use_container_width=True)
        
    with c_err2:
        st.markdown("#### 3. Variación Porcentual en Errores Estándar por Efecto HAC (Δ%)")
        fig_delta = go.Figure()
        
        fig_delta.add_trace(go.Bar(
            x=[f"{r['Activo']} vs {r['Mercado']}" for _, r in comparison_df.iterrows()],
            y=comparison_df["Δ% EE Beta"],
            name="Δ% en EE(Beta)",
            marker_color=np.where(comparison_df["Δ% EE Beta"] >= 0, "#EF4444", "#10B981"),
            text=[f"{v:+.1f}%" for v in comparison_df["Δ% EE Beta"]],
            textposition="outside"
        ))
        fig_delta.add_hline(y=0, line_color="#0F172A", line_width=1.2)
        
        fig_delta.update_layout(
            yaxis_title="Cambio en Error Estándar (%)",
            template="plotly_white",
            height=430,
            showlegend=False
        )
        st.plotly_chart(fig_delta, use_container_width=True)
        
    # 3. Panel Didáctico Explicativo
    st.markdown("""
    <div class="analysis-box">
        <b>💡 Hallazgo Econométrico Clave:</b>
        <ul>
            <li><b>En Bitcoin (BTC):</b> El error estándar de Beta con HAC <b>aumenta dramáticamente (+80.0% frente a NCI y +55.4% frente a S&P 500)</b>. Esto ocurre porque los retornos de las criptomonedas exhiben una fuerte persistencia en su volatilidad y colas pesadas. Si usáramos OLS convencional sin HAC, <b>subestimaríamos el error estándar</b>, inflando artificialmente la precisión estadística del modelo y arriesgando falsos positivos.</li>
            <li><b>En Gossamer Bio (GOSS):</b> La corrección HAC reduce ligeramente el error estándar de Beta frente a NBI (-13.2%) y S&P 500 (-5.1%), mientras que se mantiene prácticamente neutral frente a XBI (+1.7%). Esto demuestra la importancia de ajustar por autocorrelación cuando se tienen choques discretos en activos biotecnológicos.</li>
            <li><b>Invarianza de Coeficientes:</b> En todos los casos, los estimadores puntuales de <b>Alfa (α)</b>, <b>Beta (β)</b> y el <b>R²</b> son matemáticamente idénticos entre OLS y HAC. La matriz de Newey-West actúa exclusivamente sobre la varianza asintótica de los estimadores, preservando el ajuste insesgado de MCO.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PESTAÑA 2: VISUALIZACIONES Y GRÁFICAS DE MODELADO (SUSTENTACIÓN)
# -----------------------------------------------------------------------------
with tab_graphs:
    st.markdown("### 📊 Gráficas de Análisis y Sustentación Econométrica")
    st.caption("Visualizaciones cuantitativas extraídas directamente de los notebooks para respaldar el ajuste, la sensibilidad y el comportamiento de los residuos.")
    
    # 1. Gráficas de Dispersión CAPM y Recta de Regresión (SML)
    st.markdown('<div class="chart-title">1. Dispersión CAPM y Recta de Regresión de Mercado</div>', unsafe_allow_html=True)
    
    # Seleccionar qué mercados graficar según activo
    assets_to_plot = [active_ticker] if not is_consolidated else ["GOSS", "BTC"]
    
    for a_tick in assets_to_plot:
        bench_list = list(CAPM_SPECS[a_tick]["benchmarks"].keys())
        if not is_consolidated and selected_bench != "Todos los Benchmarks":
            bench_list = [selected_bench]
            
        cols_sml = st.columns(len(bench_list))
        for idx_b, b_tick in enumerate(bench_list):
            with cols_sml[idx_b]:
                data_sml = datasets_dict[a_tick][b_tick]
                model_sml = models_dict[a_tick][b_tick]["HAC"]
                
                fig_sml = px.scatter(
                    data_sml,
                    x="excess_market",
                    y="excess_asset",
                    labels={"excess_market": f"Exceso Retorno {b_tick}", "excess_asset": f"Exceso Retorno {a_tick}"},
                    title=f"{a_tick} vs {b_tick} (R² = {model_sml.rsquared*100:.1f}%)",
                    opacity=0.35
                )
                fig_sml.update_traces(marker=dict(color="#3B82F6", size=5))
                
                # Recta MCO
                x_vals = np.linspace(data_sml["excess_market"].min(), data_sml["excess_market"].max(), 100)
                y_vals = model_sml.params["const"] + model_sml.params["excess_market"] * x_vals
                
                fig_sml.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=f"y = {model_sml.params['const']:.5f} + {model_sml.params['excess_market']:.3f}x",
                    line=dict(color="#DC2626", width=2.5)
                ))
                fig_sml.add_hline(y=0, line_color="gray", line_width=0.8)
                fig_sml.add_vline(x=0, line_color="gray", line_width=0.8)
                
                fig_sml.update_layout(template="plotly_white", height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_sml, use_container_width=True)
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Beta Normal vs Beta Bajista (Downside Beta) y Poder Explicativo R2
    c_down, c_r2 = st.columns([1.2, 1.0])
    
    with c_down:
        st.markdown('<div class="chart-title">2. Asimetría: Beta Normal vs Beta Bajista (Downside Beta)</div>', unsafe_allow_html=True)
        fig_beta_down = go.Figure()
        
        comp_subset = comparison_df if is_consolidated else comparison_df[comparison_df["Activo"] == active_ticker]
        labels = [f"{r['Activo']} vs {r['Mercado']}" for _, r in comp_subset.iterrows()]
        
        fig_beta_down.add_trace(go.Bar(
            x=labels,
            y=comp_subset["Beta (β)"],
            name="Beta Normal (Muestra Completa)",
            marker_color="#2563EB"
        ))
        fig_beta_down.add_trace(go.Bar(
            x=labels,
            y=comp_subset["Beta Bajista"],
            name="Beta Bajista (Mercado a la baja: Rm < 0)",
            marker_color="#DC2626"
        ))
        fig_beta_down.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Beta = 1.0")
        
        fig_beta_down.update_layout(
            barmode="group",
            yaxis_title="Sensibilidad (Beta)",
            template="plotly_white",
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_beta_down, use_container_width=True)
        
    with c_r2:
        st.markdown('<div class="chart-title">3. Poder Explicativo del CAPM (R² por Benchmark)</div>', unsafe_allow_html=True)
        fig_r2 = px.bar(
            comp_subset,
            x=labels,
            y="R²",
            color="Activo",
            color_discrete_map={"GOSS": "#EF4444", "BTC": "#F59E0B"},
            text=[f"{v*100:.1f}%" for v in comp_subset["R²"]],
            title="Porcentaje de Varianza Explicada por Mercado"
        )
        fig_r2.update_traces(textposition="outside")
        fig_r2.update_layout(
            yaxis_title="Coeficiente R²",
            template="plotly_white",
            height=430,
            showlegend=False
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Diagnóstico Visual de Residuos OLS (Justificación de HAC)
    st.markdown('<div class="chart-title">4. Diagnóstico de Residuos OLS: Justificación Empírica de Newey-West HAC</div>', unsafe_allow_html=True)
    
    fig_resid = go.Figure()
    for a_tick in assets_to_plot:
        bench_list = list(CAPM_SPECS[a_tick]["benchmarks"].keys())
        if not is_consolidated and selected_bench != "Todos los Benchmarks":
            bench_list = [selected_bench]
            
        for b_tick in bench_list:
            mod_data = datasets_dict[a_tick][b_tick]
            mod_ols = models_dict[a_tick][b_tick]["OLS"]
            
            fig_resid.add_trace(go.Scatter(
                x=mod_data["Date"],
                y=mod_ols.resid,
                mode="lines",
                name=f"Residuos {a_tick} vs {b_tick}",
                line=dict(width=1.2),
                opacity=0.8
            ))
            
    fig_resid.add_hline(y=0, line_color="#0F172A", line_width=1.2)
    fig_resid.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Residuo MCO (ε̂_t)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_resid, use_container_width=True)

# -----------------------------------------------------------------------------
# PESTAÑA 3: TABLAS FINALES INTEGRALES (EXTRAÍDAS DE LOS NOTEBOOKS)
# -----------------------------------------------------------------------------
with tab_tables:
    st.markdown("### 📋 Tablas Finales Integrales Extraídas de los Cuadernos")
    st.caption("Tablas idénticas a las generadas y exportadas en `CAPM_GOSS.ipynb` y `CAPM_BTC.ipynb`.")
    
    # 1. Tabla Final de Coeficientes CAPM
    st.markdown("#### 1. Tabla de Resultados CAPM (Alfa, Beta, R² y Observaciones)")
    cols_capm_simple = ["Activo", "Mercado", "Modelo", "Alfa (diario)", "Alfa (anual)", "Beta", "R²", "p-value Alfa", "p-value Beta", "N"]
    st.dataframe(
        results_df[cols_capm_simple].style.format({
            "Alfa (diario)": "{:.6f}",
            "Alfa (anual)": "{:+.2%}",
            "Beta": "{:.4f}",
            "R²": "{:.4f}",
            "p-value Alfa": "{:.4f}",
            "p-value Beta": "{:.4f}",
            "N": "{:.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Tabla Final Integral: CAPM + Métricas de Riesgo y Downside Beta
    st.markdown("#### 2. Tabla Final Integral (CAPM + Riesgo + Beta Bajista)")
    cols_integral = [
        "Activo", "Mercado", "Modelo", "Alfa (diario)", "Beta", "Beta bajista",
        "R²", "Sharpe", "Sortino", "Máximo Drawdown", "VaR 95%", "CVaR 95%",
        "p-value Alfa", "p-value Beta", "N"
    ]
    st.dataframe(
        results_df[cols_integral].style.format({
            "Alfa (diario)": "{:.6f}",
            "Beta": "{:.4f}",
            "Beta bajista": "{:.4f}",
            "R²": "{:.4f}",
            "Sharpe": "{:.4f}",
            "Sortino": "{:.4f}",
            "Máximo Drawdown": "{:.2%}",
            "VaR 95%": "{:.2%}",
            "CVaR 95%": "{:.2%}",
            "p-value Alfa": "{:.4f}",
            "p-value Beta": "{:.4f}",
            "N": "{:.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Reporte Econométrico Detallado (OLS Summary)
    st.markdown("#### 3. Reporte Econométrico OLS Completo (Statsmodels Summary)")
    
    sel_asset_rep = st.selectbox("Seleccione Activo:", options=list(models_dict.keys()), key="rep_asset")
    sel_bench_rep = st.selectbox("Seleccione Benchmark:", options=list(models_dict[sel_asset_rep].keys()), key="rep_bench")
    sel_model_type = st.radio("Especificación:", options=["HAC (Newey-West)", "OLS Clásico (Sin HAC)"], horizontal=True, key="rep_type")
    
    chosen_mod = models_dict[sel_asset_rep][sel_bench_rep]["HAC" if "HAC" in sel_model_type else "OLS"]
    st.text(str(chosen_mod.summary()))
    
    # Botón de Descarga
    st.markdown("---")
    csv_bytes = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Tabla Final Integral en CSV",
        data=csv_bytes,
        file_name="CAPM_tabla_final_integral_consolidada.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# PESTAÑA 4: INTERPRETACIÓN Y SUSTENTACIÓN ACADÉMICA
# -----------------------------------------------------------------------------
with tab_theory:
    st.markdown("### 💡 Interpretación y Sustentación de Resultados Econométricos")
    st.caption("Argumentación teórica y conclusiones listas para defender metodológicamente los modelos.")
    
    st.markdown("""
    <div class="analysis-box">
        <h4>1. ¿Por qué es fundamental comparar CON HAC vs SIN HAC?</h4>
        <ul>
            <li>En finanzas empíricas, las series de retornos bursátiles y de criptoactivos sufren frecuentemente de <b>volatilidad agrupada (clustering)</b> y <b>autocorrelación residual</b>, violando el supuesto de homocedasticidad e independencia de MCO (Gauss-Markov).</li>
            <li>Bajo estas condiciones, los estimadores MCO de los coeficientes (<b>Alfa y Beta</b>) siguen siendo <b>insesgados y consistentes</b>, pero la fórmula clásica de la varianza <code>σ²(X'X)⁻¹</code> queda sesgada.</li>
            <li>El estimador de <b>Newey-West (HAC)</b> corrige esta distorsión aplicando una matriz de covarianza robusta con un kernel Bartlett de rezagos (en nuestro caso, 5 rezagos). Al hacerlo, proporciona <b>errores estándar válidos asintóticamente</b>, permitiendo contrastes de hipótesis (t-stats y p-valores) confiables y defendibles.</li>
        </ul>
    </div>
    
    <div class="analysis-box">
        <h4>2. Hallazgos Específicos para Gossamer Bio (GOSS):</h4>
        <ul>
            <li><b>Beta Moderada a Agresiva:</b> Frente a los benchmarks biotecnológicos <b>XBI (β = 1.12)</b> y <b>NBI (β = 1.57)</b>, y frente al mercado general <b>S&P 500 (β = 1.61)</b>.</li>
            <li><b>Asimetría Bajista Severa:</b> En todos los casos, el <b>Downside Beta es significativamente mayor que el Beta normal</b> (frente a S&P 500 pasa de <b>1.61 a 2.09</b>). Esto demuestra que en días de caída de mercado, GOSS amplifica las pérdidas al doble del mercado general.</li>
            <li><b>Bajo R² (< 8%):</b> Más del <b>92% de la varianza de GOSS no proviene del mercado ni del sector</b>, sino de su riesgo específico idiosincrático (resultados de ensayos clínicos y viabilidad de medicamentos).</li>
        </ul>
    </div>
    
    <div class="analysis-box">
        <h4>3. Hallazgos Específicos para Bitcoin (BTC):</h4>
        <ul>
            <li><b>Elevado Poder Explicativo frente a NCI (R² = 79.1%):</b> El índice cripto NCI explica casi el 80% del comportamiento de BTC con un <b>Beta de 0.78</b>.</li>
            <li><b>Beta frente a S&P 500 (β = 1.20, R² = 18.0%):</b> Bitcoin se comporta como un activo agresivo frente a la renta variable tradicional, con un <b>Downside Beta de 1.38</b> en fases bajistas.</li>
            <li><b>Efecto Masivo de HAC:</b> El error estándar de Beta frente a NCI se expande un <b>+80.0%</b> (de 0.0114 a 0.0206) debido a la fuerte heterocedasticidad del mercado cripto, probando que OLS convencional sobrestimaba drásticamente la certidumbre estadística.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

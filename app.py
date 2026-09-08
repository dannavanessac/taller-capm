import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from data_loader import (
    ASSET_METADATA,
    load_extracted_stats,
    load_all_series,
    calculate_kpis,
    generate_financial_diagnosis
)

# -----------------------------------------------------------------------------
# Configuración de Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tablero de Análisis y Comportamiento de Activos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 750;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.08rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #10B981;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-sub-neg {
        font-size: 0.85rem;
        color: #EF4444;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-sub-neutral {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 4px;
    }
    .conclusion-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        font-size: 1.02rem;
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
    .badge-crypto { background-color: #FEF3C7; color: #92400E; }
    .badge-stock { background-color: #E0E7FF; color: #3730A3; }
    .badge-benchmark { background-color: #D1FAE5; color: #065F46; }
    .chart-section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Carga Directa y Confiable de Datos
# -----------------------------------------------------------------------------
stats_data = load_extracted_stats()
series_data = load_all_series()

# -----------------------------------------------------------------------------
# Barra Lateral: Filtros
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=65)
st.sidebar.title("Filtros de Análisis")

# 1. Filtro de Selección de Activo
asset_options = ["🌐 Vista Comparativa General (Todos)"] + list(ASSET_METADATA.keys())
selected_option = st.sidebar.selectbox(
    "Seleccione el Activo a Visualizar:",
    options=asset_options,
    index=1, # Default: BTC
    help="Seleccione un activo individual para ver su comportamiento detallado o la vista comparativa general."
)

is_comparison_view = (selected_option == "🌐 Vista Comparativa General (Todos)")
current_ticker = None if is_comparison_view else selected_option

# 2. Filtro de Rango Temporal
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Rango Temporal")

all_min_date = min([df['Date_Parsed'].min() for df in series_data.values()]).date()
all_max_date = max([df['Date_Parsed'].max() for df in series_data.values()]).date()

date_range = st.sidebar.date_input(
    "Filtrar Período:",
    value=(all_min_date, all_max_date),
    min_value=all_min_date,
    max_value=all_max_date
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = all_min_date, all_max_date

scale_option = st.sidebar.radio(
    "Escala en Gráficas de Precio:",
    options=["Lineal", "Logarítmica"],
    index=0,
    help="Escala logarítmica recomendada para evaluar activos con movimientos porcentuales muy amplios."
)
is_log_scale = (scale_option == "Logarítmica")

st.sidebar.markdown("---")
st.sidebar.caption("📊 **Análisis de Activos**: Tablero para describir el comportamiento financiero individual e histórico de stocks y benchmarks.")

# -----------------------------------------------------------------------------
# Filtrar Datos por Fecha
# -----------------------------------------------------------------------------
filtered_series = {}
for tick, df in series_data.items():
    mask = (df['Date_Parsed'].dt.date >= start_date) & (df['Date_Parsed'].dt.date <= end_date)
    filtered_series[tick] = df[mask].reset_index(drop=True)

color_map = {
    "BTC": "#F7931A",
    "GOSS": "#EF4444",
    "NBI": "#3B82F6",
    "NCI": "#8B5CF6",
    "S&P500": "#10B981",
    "XBI": "#06B6D4"
}

# -----------------------------------------------------------------------------
# VISTA 1: VISTA COMPARATIVA GENERAL (TODOS LOS ACTIVOS)
# -----------------------------------------------------------------------------
if is_comparison_view:
    st.markdown('<div class="main-header">🌐 Tablero Comparativo Multi-Activo</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Análisis transversal de rendimiento, volatilidad y correlación | Período: <b>{start_date.strftime("%d/%m/%Y")}</b> al <b>{end_date.strftime("%d/%m/%Y")}</b></div>', unsafe_allow_html=True)
    
    # 1. Gráfica de Rendimiento Acumulado Normalizado (Base 100)
    st.markdown('<div class="chart-section-title">1. Rendimiento Relativo Acumulado Normalizado (Base $100)</div>', unsafe_allow_html=True)
    fig_comp = go.Figure()
    
    for tick, df_f in filtered_series.items():
        if df_f.empty:
            continue
        norm_series = (df_f['Close_Clean'] / df_f['Close_Clean'].iloc[0]) * 100.0
        fig_comp.add_trace(go.Scatter(
            x=df_f['Date_Parsed'],
            y=norm_series,
            mode='lines',
            name=f"{tick} - {ASSET_METADATA[tick]['category']}",
            line=dict(color=color_map.get(tick, '#000000'), width=2.5),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Valor Base 100: $%{y:,.2f}<extra></extra>"
        ))
        
    fig_comp.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Valor de la Inversión Normalizada ($)",
        yaxis_type="log" if is_log_scale else "linear",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        height=540,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # 2. Resumen Métricas y Scatter Riesgo vs Retorno
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    
    summary_rows = []
    for tick, df_f in filtered_series.items():
        if df_f.empty or len(df_f) < 5:
            continue
        meta = ASSET_METADATA[tick]
        k = calculate_kpis(df_f, is_crypto=meta['is_crypto'])
        
        summary_rows.append({
            "Activo": tick,
            "Nombre Completo": meta["name"],
            "Categoría": meta["category"],
            "Precio Inicial": f"${k['p_start']:,.2f}",
            "Precio Final": f"${k['p_end']:,.2f}",
            "Retorno Total": f"{k['total_cum_ret']*100:+.2f}%",
            "Ret. Anualizado": f"{k['ann_ret']*100:+.2f}%",
            "Vol. Anualizada": f"{k['ann_vol']*100:.2f}%",
            "Asimetría (Skew)": f"{k['skewness']:.2f}",
            "Curtosis": f"{k['kurtosis']:.2f}",
            "_raw_ret": k['ann_ret'] * 100,
            "_raw_vol": k['ann_vol'] * 100
        })
        
    df_summary = pd.DataFrame(summary_rows)
    
    with col_left:
        st.markdown('<div class="chart-section-title">2. Mapa de Riesgo vs Retorno</div>', unsafe_allow_html=True)
        if not df_summary.empty:
            fig_risk_ret = px.scatter(
                df_summary,
                x="_raw_vol",
                y="_raw_ret",
                text="Activo",
                color="Activo",
                color_discrete_map=color_map,
                hover_data={
                    "Nombre Completo": True,
                    "Ret. Anualizado": True,
                    "Vol. Anualizada": True,
                    "_raw_vol": False,
                    "_raw_ret": False
                }
            )
            fig_risk_ret.update_traces(
                textposition="top center",
                marker=dict(size=18, opacity=0.85, line=dict(width=1.5, color='#1E293B'))
            )
            fig_risk_ret.update_layout(
                xaxis_title="Volatilidad Anualizada (%) [Riesgo]",
                yaxis_title="Retorno Anualizado (%) [Rendimiento]",
                template="plotly_white",
                height=480,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_risk_ret, use_container_width=True)
            
    with col_right:
        st.markdown('<div class="chart-section-title">3. Matriz de Correlación de Retornos Diarios</div>', unsafe_allow_html=True)
        all_ret_dfs = []
        for tick, df_f in filtered_series.items():
            s_ret = df_f[['Date_Parsed', 'Log_Returns_Clean']].rename(columns={'Log_Returns_Clean': tick})
            all_ret_dfs.append(s_ret)
            
        merged_all = all_ret_dfs[0]
        for s_ret in all_ret_dfs[1:]:
            merged_all = pd.merge(merged_all, s_ret, on='Date_Parsed', how='inner')
            
        corr_matrix = merged_all.drop(columns=['Date_Parsed']).corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        fig_corr.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # 4. Tabla Consolidada de Desempeño
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-section-title">4. Tabla Comparativa Consolidada de Activos</div>', unsafe_allow_html=True)
    cols_display = ["Activo", "Nombre Completo", "Categoría", "Precio Inicial", "Precio Final", "Retorno Total", "Ret. Anualizado", "Vol. Anualizada", "Asimetría (Skew)", "Curtosis"]
    st.dataframe(df_summary[cols_display], use_container_width=True, hide_index=True)
    
    # 5. Conclusiones Generales
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-section-title">5. 💡 Conclusiones Generales sobre el Comportamiento de los Activos</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="conclusion-box">
        <b>1. Divergencia en Perfiles de Riesgo y Rentabilidad:</b>
        <ul>
            <li><b>S&P 500 (^GSPC)</b> demostró la mayor consistencia y estabilidad, combinando rentabilidad positiva sostenida (+70.7% acumulado) con la menor volatilidad del grupo (17.2% anual).</li>
            <li><b>Criptoactivos (BTC y NCI)</b> registraron alta apreciación pero con oscilaciones estructuralmente extremas (volatilidad superior al 50-65% anual) y una alta correlación mutua (0.89).</li>
            <li><b>Gossamer Bio (GOSS)</b> ilustra el riesgo de una acción biotecnológica individual en etapa clínica, sufriendo una contracción superior al 98% con extrema curtosis (161.3).</li>
            <li><b>Índices Sectoriales Biotecnológicos (XBI y NBI)</b> mantuvieron retornos acumulados positivos (+29% a +36%), demostrando cómo la diversificación de un índice o ETF amortigua el colapso de activos individuales como GOSS.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VISTA 2: ANÁLISIS INDIVIDUAL POR ACTIVO
# -----------------------------------------------------------------------------
else:
    info = ASSET_METADATA[current_ticker]
    df_active = filtered_series[current_ticker]
    
    badge_class = "badge-crypto" if info["is_crypto"] else ("badge-stock" if "Stock" in info["category"] else "badge-benchmark")
    st.markdown(f'<div class="main-header">{info["name"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge {badge_class}">{info["category"]}</span> <span style="color:#64748B; font-size:1.0rem;">{info["description"]}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if df_active.empty or len(df_active) < 5:
        st.warning("No hay suficientes datos disponibles para el rango de fechas seleccionado.")
        st.stop()
        
    kpis = calculate_kpis(df_active, is_crypto=info["is_crypto"])
    
    # 4 Tarjetas de Métricas Clave (Sin Sharpe, Sin Max Drawdown, Sin VaR)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precio Actual</div>
            <div class="metric-value">${kpis['p_end']:,.2f}</div>
            <div class="metric-sub-neutral">Precio Inicial: ${kpis['p_start']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        ret_c = "metric-sub" if kpis['total_cum_ret'] >= 0 else "metric-sub-neg"
        color_val = "#10B981" if kpis['total_cum_ret'] >= 0 else "#EF4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Retorno Total Acumulado</div>
            <div class="metric-value" style="color:{color_val};">{kpis['total_cum_ret']*100:+.2f}%</div>
            <div class="{ret_c}">Anualizado: {kpis['ann_ret']*100:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volatilidad Anualizada</div>
            <div class="metric-value">{kpis['ann_vol']*100:.2f}%</div>
            <div class="metric-sub-neutral">Desviación Diaria: {kpis['std_daily']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rango de Precios del Período</div>
            <div class="metric-value" style="font-size: 1.35rem;">${kpis['p_min']:,.2f} - ${kpis['p_max']:,.2f}</div>
            <div class="metric-sub-neutral">Mínimo y Máximo Registrado</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Pestañas de Navegación del Activo
    tab_graphs, tab_tables, tab_diagnosis = st.tabs([
        "📊 Visualizaciones de Comportamiento del Activo",
        "📋 Tablas Extraídas y Datos",
        "💡 Diagnóstico y Conclusiones del Activo"
    ])
    
    # -------------------------------------------------------------------------
    # PESTAÑA 1: VISUALIZACIONES DE COMPORTAMIENTO (6 GRÁFICAS AMPLIAS)
    # -------------------------------------------------------------------------
    with tab_graphs:
        # Gráfica 1: Precios y Medias Móviles (Ancho Completo, Grande)
        st.markdown('<div class="chart-section-title">1. Evolución Histórica de Precios y Medias Móviles (SMA 20, SMA 50, SMA 200)</div>', unsafe_allow_html=True)
        fig_price = go.Figure()
        
        fig_price.add_trace(go.Scatter(
            x=df_active['Date_Parsed'],
            y=df_active['Close_Clean'],
            name=f"Precio de Cierre ({current_ticker})",
            line=dict(color="#1D4ED8", width=2.5),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Precio: $%{y:,.2f}<extra></extra>"
        ))
        fig_price.add_trace(go.Scatter(
            x=df_active['Date_Parsed'],
            y=df_active['SMA_20'],
            name="Media Móvil 20 días (Corto Plazo)",
            line=dict(color="#06B6D4", width=1.6)
        ))
        fig_price.add_trace(go.Scatter(
            x=df_active['Date_Parsed'],
            y=df_active['SMA_50'],
            name="Media Móvil 50 días (Mediano Plazo)",
            line=dict(color="#F59E0B", width=1.8, dash="dash")
        ))
        fig_price.add_trace(go.Scatter(
            x=df_active['Date_Parsed'],
            y=df_active['SMA_200'],
            name="Media Móvil 200 días (Largo Plazo)",
            line=dict(color="#DC2626", width=2.0, dash="dot")
        ))
        
        fig_price.update_layout(
            yaxis_type="log" if is_log_scale else "linear",
            yaxis_title=f"Precio ($ {'Log' if is_log_scale else ''})",
            xaxis_title="Fecha",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Gráfica 2: Volumen y Liquidez (Ancho Completo, Seguro contra fallos)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-section-title">2. Dinámica de Negociación y Volumen de Mercado</div>', unsafe_allow_html=True)
        
        # Comprobar volumen de forma segura
        has_valid_vol = False
        if 'Volume_Clean' in df_active.columns:
            clean_vol_series = df_active['Volume_Clean'].dropna()
            if len(clean_vol_series) > 0 and clean_vol_series.sum() > 0:
                has_valid_vol = True
                
        if has_valid_vol and info["has_volume"]:
            # Calcular SMA de volumen en tiempo real de forma segura
            vol_sma20 = df_active['Volume_Clean'].rolling(20, min_periods=1).mean()
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=df_active['Date_Parsed'],
                y=df_active['Volume_Clean'],
                mode='lines',
                fill='tozeroy',
                name='Volumen Diario Transado',
                line=dict(color='#3B82F6', width=1.3),
                fillcolor='rgba(59, 130, 246, 0.20)',
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Volumen: %{y:,.0f}<extra></extra>"
            ))
            fig_vol.add_trace(go.Scatter(
                x=df_active['Date_Parsed'],
                y=vol_sma20,
                mode='lines',
                name='Media Móvil de Volumen (20 días)',
                line=dict(color='#1E3A8A', width=2.5),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Promedio 20d: %{y:,.0f}<extra></extra>"
            ))
            
            fig_vol.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Volumen Negociado",
                template="plotly_white",
                height=450,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            # Para NCI o activos sin volumen transaccional
            st.info("ℹ️ **Nota sobre NCI**: El archivo de datos de Nasdaq Crypto Index no contiene columna de volumen transaccional directo. A continuación se presenta el **Rango de Precios Intradiario (High - Low)**:")
            
            if 'High_Clean' in df_active.columns and 'Low_Clean' in df_active.columns:
                spread_series = df_active['High_Clean'] - df_active['Low_Clean']
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=df_active['Date_Parsed'],
                    y=spread_series,
                    mode='lines',
                    fill='tozeroy',
                    name='Rango Intradiario (High - Low)',
                    line=dict(color='#8B5CF6', width=1.6),
                    fillcolor='rgba(139, 92, 246, 0.20)'
                ))
                fig_spread.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Diferencial Intradiario ($)",
                    template="plotly_white",
                    height=420,
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_spread, use_container_width=True)
            else:
                fig_cum = px.line(
                    df_active,
                    x='Date_Parsed',
                    y=df_active['Cumulative_Return']*100,
                    title=f"Retorno Acumulado (%) - {current_ticker}"
                )
                fig_cum.update_traces(line_color="#8B5CF6", line_width=2.2)
                fig_cum.update_layout(template="plotly_white", height=420)
                st.plotly_chart(fig_cum, use_container_width=True)
                
        # Gráficas 3 y 4: Retornos Diarios y Volatilidad Móvil (2 Columnas Grandes)
        st.markdown("<br>", unsafe_allow_html=True)
        c_ret, c_vol = st.columns(2)
        
        with c_ret:
            st.markdown('<div class="chart-section-title">3. Retornos Diarios y Clústeres de Volatilidad</div>', unsafe_allow_html=True)
            fig_ret = go.Figure()
            
            colors = ['#10B981' if v >= 0 else '#EF4444' for v in df_active['Log_Returns_Clean'].fillna(0)]
            fig_ret.add_trace(go.Bar(
                x=df_active['Date_Parsed'],
                y=df_active['Log_Returns_Clean'],
                marker_color=colors,
                name="Retorno Diario",
                opacity=0.75,
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Retorno: %{y:+.2%}<extra></extra>"
            ))
            
            two_std = 2 * kpis['std_daily']
            fig_ret.add_hline(y=two_std, line_dash="dash", line_color="#F59E0B", annotation_text="+2σ (Alerta)")
            fig_ret.add_hline(y=-two_std, line_dash="dash", line_color="#F59E0B", annotation_text="-2σ (Alerta)")
            
            fig_ret.update_layout(
                yaxis_title="Retorno Logarítmico Diario",
                xaxis_title="Fecha",
                template="plotly_white",
                height=460,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_ret, use_container_width=True)
            
        with c_vol:
            st.markdown('<div class="chart-section-title">4. Volatilidad Móvil Anualizada (30 y 90 Días)</div>', unsafe_allow_html=True)
            fig_volat = go.Figure()
            
            fig_volat.add_trace(go.Scatter(
                x=df_active['Date_Parsed'],
                y=df_active['Rolling_Vol_30d'] * 100,
                name="Volatilidad Móvil 30 días (Reactiva)",
                line=dict(color="#EC4899", width=2.0)
            ))
            fig_volat.add_trace(go.Scatter(
                x=df_active['Date_Parsed'],
                y=df_active['Rolling_Vol_90d'] * 100,
                name="Volatilidad Móvil 90 días (Estructural)",
                line=dict(color="#8B5CF6", width=2.2)
            ))
            fig_volat.add_hline(
                y=kpis['ann_vol'] * 100,
                line_dash="dot",
                line_color="#475569",
                annotation_text=f"Promedio: {kpis['ann_vol']*100:.1f}%"
            )
            
            fig_volat.update_layout(
                yaxis_title="Volatilidad Anualizada (%)",
                xaxis_title="Fecha",
                template="plotly_white",
                height=460,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_volat, use_container_width=True)
            
        # Gráficas 5 y 6: Distribución de Retornos y Drawdowns (2 Columnas Grandes)
        st.markdown("<br>", unsafe_allow_html=True)
        c_dist, c_dd = st.columns(2)
        
        with c_dist:
            st.markdown('<div class="chart-section-title">5. Distribución de Retornos vs Curva Normal Teórica</div>', unsafe_allow_html=True)
            clean_rets = df_active['Log_Returns_Clean'].dropna()
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=clean_rets,
                histnorm='probability density',
                nbinsx=70,
                name="Distribución Empírica de Retornos",
                marker_color="#3B82F6",
                opacity=0.65
            ))
            
            x_norm = np.linspace(clean_rets.min(), clean_rets.max(), 250)
            y_norm = stats.norm.pdf(x_norm, clean_rets.mean(), clean_rets.std())
            fig_dist.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Gaussiana Teórica N(μ, σ)',
                line=dict(color="#DC2626", width=2.5)
            ))
            
            fig_dist.update_layout(
                title=f"Asimetría (Skewness): <b>{kpis['skewness']:.2f}</b> | Curtosis: <b>{kpis['kurtosis']:.2f}</b>",
                xaxis_title="Retorno Logarítmico",
                yaxis_title="Densidad de Probabilidad",
                template="plotly_white",
                height=460,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with c_dd:
            st.markdown('<div class="chart-section-title">6. Curva de Caídas Históricas (Underwater / Drawdown)</div>', unsafe_allow_html=True)
            fig_dd = go.Figure()
            
            fig_dd.add_trace(go.Scatter(
                x=df_active['Date_Parsed'],
                y=df_active['Drawdown'] * 100,
                fill='tozeroy',
                name="Caída desde Máximo Previo (%)",
                line=dict(color="#EF4444", width=1.6),
                fillcolor="rgba(239, 68, 68, 0.25)",
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Caída: %{y:.2f}%<extra></extra>"
            ))
            fig_dd.add_hline(
                y=kpis['max_drawdown'] * 100,
                line_dash="dash",
                line_color="#991B1B",
                annotation_text=f"Máxima Caída: {kpis['max_drawdown']*100:.2f}%"
            )
            
            fig_dd.update_layout(
                yaxis_title="Caída desde Máximo Anterior (%)",
                xaxis_title="Fecha",
                template="plotly_white",
                height=460,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    # -------------------------------------------------------------------------
    # PESTAÑA 2: TABLAS EXTRAÍDAS Y DATOS HISTÓRICOS
    # -------------------------------------------------------------------------
    with tab_tables:
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown(f"#### 1. Tabla de Estadísticas Extraída de la Carpeta `{info['folder']}`")
            st.caption(f"Archivo fuente extraído: `{info['folder']}/{current_ticker.lower() if current_ticker != 'S&P500' else 'gspc'}_stats.csv`")
            
            raw_stats_df = stats_data.get(current_ticker, pd.DataFrame())
            if not raw_stats_df.empty:
                display_stats = raw_stats_df.copy()
                formatted_vals = []
                for idx, r in display_stats.iterrows():
                    val = r["Valor Numérico"]
                    met = str(r["Métrica"]).lower()
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        if "retorno" in met or "volatilidad" in met:
                            formatted_vals.append(f"{val*100:.4f}% ({val:.6f})")
                        else:
                            formatted_vals.append(f"{val:.6f}")
                    else:
                        formatted_vals.append(str(r["Valor Original"]))
                display_stats["Valor Formateado"] = formatted_vals
                st.dataframe(display_stats[["Métrica", "Valor Formateado", "Valor Original"]], use_container_width=True, hide_index=True)
            else:
                st.info("No se encontró archivo de estadísticas para este activo.")
                
        with col_t2:
            st.markdown(f"#### 2. Tabla de Métricas Cuantitativas Calculadas (Rango Filtrado)")
            st.caption(f"Período evaluado: {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}")
            
            calc_metrics = [
                {"Métrica": "Precio de Cierre Inicial", "Valor": f"${kpis['p_start']:,.2f}"},
                {"Métrica": "Precio de Cierre Final", "Valor": f"${kpis['p_end']:,.2f}"},
                {"Métrica": "Precio Máximo Registrado", "Valor": f"${kpis['p_max']:,.2f}"},
                {"Métrica": "Precio Mínimo Registrado", "Valor": f"${kpis['p_min']:,.2f}"},
                {"Métrica": "Retorno Acumulado Total", "Valor": f"{kpis['total_cum_ret']*100:+.2f}%"},
                {"Métrica": "Retorno Promedio Diario", "Valor": f"{kpis['mean_daily']*100:+.4f}%"},
                {"Métrica": "Retorno Anualizado (Log)", "Valor": f"{kpis['ann_ret']*100:+.2f}%"},
                {"Métrica": "Desviación Estándar Diaria", "Valor": f"{kpis['std_daily']*100:.4f}%"},
                {"Métrica": "Volatilidad Anualizada", "Valor": f"{kpis['ann_vol']*100:.2f}%"},
                {"Métrica": "Varianza Diaria", "Valor": f"{kpis['variance_daily']:.6f}"},
                {"Métrica": "Asimetría (Skewness)", "Valor": f"{kpis['skewness']:.4f}"},
                {"Métrica": "Curtosis de Retornos Diarios", "Valor": f"{kpis['kurtosis']:.4f}"},
                {"Métrica": "Máxima Caída (Maximum Drawdown)", "Valor": f"{kpis['max_drawdown']*100:.2f}%"},
                {"Métrica": "Total de Observaciones Diarias", "Valor": f"{kpis['count']} días"}
            ]
            st.dataframe(pd.DataFrame(calc_metrics), use_container_width=True, hide_index=True)
            
        st.markdown("---")
        st.markdown(f"#### 3. Explorador de Serie Histórica Limpia (`df_{current_ticker.lower() if current_ticker != 'S&P500' else 'gspc'}.csv`)")
        
        export_cols = ['Date', 'Close_Clean', 'Log_Returns_Clean', 'Cumulative_Return', 'SMA_20', 'SMA_50', 'SMA_200', 'Drawdown']
        if info["has_volume"] and 'Volume_Clean' in df_active.columns and not df_active['Volume_Clean'].isna().all():
            export_cols.insert(2, 'Volume_Clean')
            
        view_df = df_active[export_cols].rename(columns={
            'Date': 'Fecha',
            'Close_Clean': 'Precio Cierre ($)',
            'Volume_Clean': 'Volumen',
            'Log_Returns_Clean': 'Retorno Log Diario',
            'Cumulative_Return': 'Retorno Acumulado',
            'SMA_20': 'SMA 20 ($)',
            'SMA_50': 'SMA 50 ($)',
            'SMA_200': 'SMA 200 ($)',
            'Drawdown': 'Drawdown'
        })
        
        st.dataframe(view_df.head(150), use_container_width=True)
        
        csv_data = view_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Descargar Datos Históricos de {current_ticker} (CSV)",
            data=csv_data,
            file_name=f"{current_ticker}_serie_historica.csv",
            mime="text/csv"
        )

    # -------------------------------------------------------------------------
    # PESTAÑA 3: DIAGNÓSTICO Y CONCLUSIONES DEL ACTIVO
    # -------------------------------------------------------------------------
    with tab_diagnosis:
        st.markdown(f"### 💡 Diagnóstico Financiero Integral: {info['name']}")
        st.caption("Conclusiones cuantitativas deducidas directamente a partir del comportamiento histórico del activo.")
        
        conclusions = generate_financial_diagnosis(current_ticker, kpis)
        
        for c in conclusions:
            st.markdown(f'<div class="conclusion-box">{c}</div>', unsafe_allow_html=True)
            
        st.markdown("#### 🎯 Resumen Ejecutivo del Comportamiento del Activo")
        if current_ticker == "GOSS":
            st.markdown("""
            * **Riesgo Específico y Pérdida Extrema:** Gossamer Bio sufrió un colapso prácticamente total (-98.38%) derivado del riesgo clínico binario característico de las biotecnológicas en fase de ensayos.
            * **Curtosis Desproporcionada (161.3):** La presencia de saltos negativos de más del 70% en un solo día explica su descomunal curtosis y la incapacidad de recuperación de la acción en los años posteriores.
            * **Implicación:** No ofrece preservación de capital y únicamente es apto para estrategias de capital de riesgo de alta especulación con límites estrictos de pérdida.
            """)
        elif current_ticker in ["BTC", "NCI"]:
            st.markdown("""
            * **Perfil de Alta Apreciación con Severos Ciclos Bajistas:** El activo demostró una capacidad excepcional de crecimiento a largo plazo, pero con correcciones intermedias superiores al -76%.
            * **Volatilidad Estructural Elevada:** La volatilidad anualizada supera ampliamente el 50%, requiriendo que los inversionistas cuenten con alta tolerancia al riesgo y un horizonte de tiempo extendido.
            """)
        elif current_ticker == "S&P500":
            st.markdown("""
            * **Estabilidad y Consistencia de Mercado:** Exhibió el comportamiento más resiliente y predecible del conjunto analizado, con un retorno acumulado del +70.7% y un control riguroso de la volatilidad (17.2% anual).
            * **Ancla de Diversificación:** Es el activo con menor drawdown (-25.4%) y menor varianza diaria entre todos los examinados.
            """)
        else: # XBI, NBI
            st.markdown("""
            * **Crecimiento Sectorial Diversificado:** A diferencia de una acción individual (como GOSS), los índices y ETFs biotecnológicos lograron retornos acumulados positivos (+29% a +36%), absorbiendo las quiebras de empresas individuales mediante la diversificación dentro de la industria médica.
            """)

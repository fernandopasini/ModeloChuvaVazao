# -*- coding: utf-8 -*-
"""
app_streamlit.py
Aplicativo Streamlit para previsão de vazão com modelos Random Forest separados por bacia.
"""

from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from funcoes_previsao import listar_sites, carregar_modelo_por_site, prever_vazao_futura

st.set_page_config(page_title="Previsão de Vazão — Random Forest", page_icon="💧", layout="wide")


@st.cache_data
def obter_sites():
    return listar_sites()


@st.cache_resource
def obter_modelo(site_key: str):
    return carregar_modelo_por_site(site_key)


def criar_tabela_chuva_recente(n_dias: int = 29) -> pd.DataFrame:
    hoje = date.today()
    datas = [hoje - timedelta(days=i) for i in range(n_dias, 0, -1)]
    return pd.DataFrame({"data": pd.to_datetime(datas), "precipitacao_mm": [0.0] * n_dias})


def criar_tabela_chuva_futura(n_dias: int = 7) -> pd.DataFrame:
    hoje = date.today()
    datas = [hoje + timedelta(days=i) for i in range(1, n_dias + 1)]
    return pd.DataFrame({"data": pd.to_datetime(datas), "precipitacao_mm": [0.0] * n_dias})


def grafico_resultados(df_resultado: pd.DataFrame, site_name: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_resultado["data"],
        y=df_resultado["precipitacao_prevista_mm"],
        name="Precipitação prevista (mm)",
        yaxis="y2",
        opacity=0.45,
    ))
    fig.add_trace(go.Scatter(
        x=df_resultado["data"],
        y=df_resultado["vazao_prevista"],
        name="Vazão prevista",
        mode="lines+markers",
    ))
    fig.update_layout(
        title=f"Previsão de vazão — {site_name}",
        xaxis_title="Data",
        yaxis=dict(title="Vazão prevista"),
        yaxis2=dict(title="Precipitação prevista (mm)", overlaying="y", side="right", autorange="reversed", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


st.title("💧 Previsão de vazão com Random Forest")
st.caption("Aplicação operacional para simular vazão futura a partir da chuva recente, da vazão atual e de cenários de precipitação futura.")

try:
    sites = obter_sites()
except Exception as e:
    st.error("Não foi possível carregar o manifesto dos modelos.")
    st.exception(e)
    st.stop()

col_esq, col_dir = st.columns([0.34, 0.66])

with col_esq:
    st.subheader("1. Seleção do modelo")
    site_key = st.selectbox("Selecione a bacia/série de vazão", options=list(sites.keys()), format_func=lambda k: sites[k])

    try:
        model, meta = obter_modelo(site_key)
    except Exception as e:
        st.error("Não foi possível carregar o modelo selecionado.")
        st.exception(e)
        st.stop()

    st.info(
        f"Modelo carregado: **{meta.get('site_name', sites[site_key])}**\n\n"
        f"Coluna de precipitação: `{meta.get('p_col')}`\n\n"
        f"Coluna de vazão: `{meta.get('q_col')}`"
    )

    st.subheader("2. Condição atual")
    q_atual = st.number_input(
        "Vazão atual / vazão observada do último dia",
        min_value=0.0,
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Esta vazão será usada como Q(t-1) no primeiro dia previsto.",
    )
    n_dias_futuros = st.number_input("Número de dias futuros para simular", min_value=1, max_value=30, value=7, step=1)

with col_dir:
    st.subheader("3. Chuva recente e chuva futura")
    st.markdown(
        "O modelo usa acumulados de precipitação de até **30 dias**. "
        "Para o primeiro dia futuro, a tabela abaixo deve conter os **29 dias anteriores**; "
        "a chuva do próprio dia previsto entra na tabela de chuva futura."
    )
    aba_recente, aba_futura = st.tabs(["Chuva recente", "Chuva futura"])
    with aba_recente:
        st.write("Informe a precipitação observada dos 29 dias anteriores ao primeiro dia de previsão.")
        chuva_recente = st.data_editor(criar_tabela_chuva_recente(29), num_rows="fixed", use_container_width=True, key="chuva_recente_editor")
    with aba_futura:
        st.write("Informe a precipitação prevista para cada dia futuro.")
        chuva_futura = st.data_editor(criar_tabela_chuva_futura(int(n_dias_futuros)), num_rows="dynamic", use_container_width=True, key=f"chuva_futura_editor_{int(n_dias_futuros)}")

st.divider()

if st.button("Simular vazão", type="primary"):
    try:
        resultado = prever_vazao_futura(model=model, meta=meta, chuva_recente=chuva_recente, chuva_futura=chuva_futura, q_atual=q_atual)
        st.subheader("Resultado da simulação")
        st.dataframe(resultado, use_container_width=True)
        grafico_resultados(resultado, meta.get("site_name", sites[site_key]))
        csv = resultado.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar resultado em CSV", data=csv, file_name=f"previsao_vazao_{site_key}.csv", mime="text/csv")
    except Exception as e:
        st.error("Não foi possível executar a simulação. Verifique os dados informados.")
        st.exception(e)

st.divider()
st.caption("Observação: para simulações de vários dias, a vazão prevista em um dia é usada como memória hidrológica para o dia seguinte. Portanto, a incerteza tende a aumentar com o horizonte de previsão.")

# -*- coding: utf-8 -*-
"""
app_streamlit.py
Aplicativo Streamlit para previsão de vazão com modelos Random Forest separados por bacia.

Versão com visual institucional, espaço para logo da UFPR, referência bibliográfica
 e link de acesso direto ao trabalho/repositório.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from funcoes_previsao import listar_sites, carregar_modelo_por_site, prever_vazao_futura

# ============================================================
# CONFIGURAÇÕES DO PROJETO — EDITE AQUI
# ============================================================
APP_TITLE = "Previsão de Vazão com Random Forest"
APP_SUBTITLE = (
    "Ferramenta operacional para simular vazão futura a partir da chuva recente, "
    "da vazão atual e de cenários de precipitação futura."
)
INSTITUTION_NAME = "Universidade Federal do Paraná — UFPR"
PROJECT_CONTEXT = "Projeto desenvolvido no âmbito da UFPR"

# Coloque o arquivo do logo no GitHub em: assets/logo_ufpr.png
# Também funciona com .jpg/.jpeg, basta alterar o caminho abaixo.
LOGO_PATH = Path("assets/logo_ufpr.png")

# Referência bibliográfica: substitua pelo texto definitivo.
BIBLIOGRAPHIC_REFERENCE = (
    "[INSERIR REFERÊNCIA BIBLIOGRÁFICA COMPLETA AQUI — autores, ano, título, "
    "instituição/periódico, DOI ou URL, se houver.]"
)

# Link de acesso direto: substitua pela URL do artigo, repositório, relatório ou página do projeto.
PROJECT_LINK_LABEL = "Acesso direto ao trabalho / repositório"
PROJECT_LINK_URL = "[INSERIR_LINK_AQUI]"

# Texto metodológico exibido no rodapé.
METHODOLOGICAL_NOTE = (
    "Para simulações de vários dias, a vazão prevista em um dia é usada como memória "
    "hidrológica para o dia seguinte. Portanto, a incerteza tende a aumentar com o horizonte "
    "de previsão. O modelo utiliza precipitação antecedente, acumulados móveis de chuva e "
    "Q(t−1) como memória hidrológica operacional."
)

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Previsão de Vazão — Random Forest",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ESTILO VISUAL
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --ufpr-blue: #003366;
        --ufpr-blue-2: #0b4f8a;
        --soft-blue: #eaf3fb;
        --soft-gray: #f7f9fb;
        --dark-text: #1f2937;
        --muted-text: #6b7280;
        --card-border: #e5e7eb;
    }

    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    .hero {
        background: linear-gradient(135deg, #003366 0%, #0b4f8a 52%, #2b7bbb 100%);
        color: white;
        padding: 1.35rem 1.55rem;
        border-radius: 18px;
        margin-bottom: 1.05rem;
        box-shadow: 0 10px 28px rgba(0, 51, 102, 0.20);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.05rem;
        line-height: 1.15;
        font-weight: 750;
    }

    .hero p {
        margin: 0.55rem 0 0 0;
        font-size: 1.02rem;
        opacity: 0.96;
    }

    .institution-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-bottom: 0.65rem;
    }

    .info-card {
        background: white;
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.0rem 1.05rem;
        box-shadow: 0 8px 20px rgba(17, 24, 39, 0.04);
        margin-bottom: 0.85rem;
    }

    .info-card h3 {
        margin-top: 0;
        margin-bottom: 0.35rem;
        font-size: 1.05rem;
        color: var(--ufpr-blue);
    }

    .small-muted {
        color: var(--muted-text);
        font-size: 0.92rem;
        line-height: 1.42;
    }

    .reference-box {
        background: #f8fafc;
        border-left: 5px solid var(--ufpr-blue);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-top: 1.2rem;
        color: var(--dark-text);
    }

    .reference-box h4 {
        margin-top: 0;
        margin-bottom: 0.45rem;
        color: var(--ufpr-blue);
    }

    .stButton > button {
        border-radius: 12px;
        padding: 0.55rem 1.2rem;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    .footer-note {
        color: var(--muted-text);
        font-size: 0.9rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CACHE
# ============================================================
@st.cache_data
def obter_sites():
    return listar_sites()


@st.cache_resource
def obter_modelo(site_key: str):
    return carregar_modelo_por_site(site_key)


# ============================================================
# FUNÇÕES AUXILIARES DA INTERFACE
# ============================================================
def criar_tabela_chuva_recente(n_dias: int = 29) -> pd.DataFrame:
    hoje = date.today()
    datas = [hoje - timedelta(days=i) for i in range(n_dias, 0, -1)]
    return pd.DataFrame({"data": pd.to_datetime(datas), "precipitacao_mm": [0.0] * n_dias})


def criar_tabela_chuva_futura(n_dias: int = 7) -> pd.DataFrame:
    hoje = date.today()
    datas = [hoje + timedelta(days=i) for i in range(1, n_dias + 1)]
    return pd.DataFrame({"data": pd.to_datetime(datas), "precipitacao_mm": [0.0] * n_dias})


def render_logo_sidebar():
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.markdown(
            """
            <div style="background:#003366;color:white;padding:1rem;border-radius:14px;text-align:center;margin-bottom:1rem;">
                <div style="font-size:1.35rem;font-weight:800;">UFPR</div>
                <div style="font-size:0.85rem;opacity:0.9;">logo_ufpr.png não encontrado</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Para exibir o logo, adicione `assets/logo_ufpr.png` ao repositório.")


def grafico_resultados(df_resultado: pd.DataFrame, site_name: str):
    d = df_resultado.copy()
    d["data"] = pd.to_datetime(d["data"])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=d["data"],
            y=d["precipitacao_prevista_mm"],
            name="Precipitação prevista (mm)",
            yaxis="y2",
            opacity=0.55,
            marker=dict(color="rgba(43, 123, 187, 0.55)"),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Precipitação: %{y:.2f} mm<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["data"],
            y=d["vazao_prevista"],
            name="Vazão prevista",
            mode="lines+markers",
            line=dict(width=3, color="#003366"),
            marker=dict(size=8, color="#003366"),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Vazão prevista: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Previsão de vazão — {site_name}",
            x=0.02,
            xanchor="left",
            font=dict(size=21),
        ),
        xaxis=dict(
            title="Data",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title="Vazão prevista",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Precipitação prevista (mm)",
            overlaying="y",
            side="right",
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
        ),
        height=560,
        margin=dict(l=50, r=55, t=90, b=45),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)


def mostrar_metricas(resultado: pd.DataFrame):
    q1 = float(resultado["vazao_prevista"].iloc[0])
    qmax = float(resultado["vazao_prevista"].max())
    qmean = float(resultado["vazao_prevista"].mean())
    ptotal = float(resultado["precipitacao_prevista_mm"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vazão no 1º dia", f"{q1:.3f}")
    col2.metric("Vazão máxima", f"{qmax:.3f}")
    col3.metric("Vazão média", f"{qmean:.3f}")
    col4.metric("Chuva total prevista", f"{ptotal:.1f} mm")


def formatar_tabela_resultados(resultado: pd.DataFrame) -> pd.DataFrame:
    out = resultado.copy()
    out["data"] = pd.to_datetime(out["data"]).dt.strftime("%d/%m/%Y")
    out = out.rename(
        columns={
            "data": "Data",
            "precipitacao_prevista_mm": "Precipitação prevista (mm)",
            "vazao_prevista": "Vazão prevista",
        }
    )
    return out


# ============================================================
# SIDEBAR
# ============================================================
render_logo_sidebar()
st.sidebar.markdown(f"### {INSTITUTION_NAME}")
st.sidebar.caption(PROJECT_CONTEXT)
st.sidebar.divider()

# ============================================================
# CABEÇALHO
# ============================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="institution-pill">{PROJECT_CONTEXT}</div>
        <h1>💧 {APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    sites = obter_sites()
except Exception as e:
    st.error("Não foi possível carregar o manifesto dos modelos.")
    st.exception(e)
    st.stop()

# ============================================================
# PAINEL PRINCIPAL
# ============================================================
col_esq, col_dir = st.columns([0.34, 0.66], gap="large")

with col_esq:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("1. Modelo")
    site_key = st.selectbox(
        "Selecione a bacia/série de vazão",
        options=list(sites.keys()),
        format_func=lambda k: sites[k],
    )

    try:
        model, meta = obter_modelo(site_key)
    except Exception as e:
        st.error("Não foi possível carregar o modelo selecionado.")
        st.exception(e)
        st.stop()

    st.success(f"Modelo carregado: {meta.get('site_name', sites[site_key])}")
    st.caption(
        f"Precipitação: `{meta.get('p_col')}`  |  "
        f"Vazão: `{meta.get('q_col')}`  |  "
        f"Q lags: `{meta.get('q_lags', [1])}`"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("2. Condição atual")
    q_atual = st.number_input(
        "Vazão atual / vazão observada do último dia",
        min_value=0.0,
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Esta vazão será usada como Q(t−1) no primeiro dia previsto.",
    )
    n_dias_futuros = st.number_input(
        "Número de dias futuros para simular",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card">
            <h3>Como interpretar</h3>
            <div class="small-muted">
                A previsão é sequencial: a vazão estimada para um dia passa a ser usada
                como memória hidrológica no dia seguinte. Para horizontes mais longos,
                a incerteza tende a aumentar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_dir:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("3. Chuva recente e chuva futura")
    st.markdown(
        "O modelo usa acumulados de precipitação de até **30 dias**. "
        "Para o primeiro dia futuro, informe os **29 dias anteriores**; "
        "a chuva do próprio dia previsto entra na tabela de chuva futura."
    )

    aba_recente, aba_futura = st.tabs(["🌧️ Chuva recente", "🔮 Chuva futura"])

    with aba_recente:
        st.write("Precipitação observada dos 29 dias anteriores ao primeiro dia de previsão.")
        chuva_recente = st.data_editor(
            criar_tabela_chuva_recente(29),
            num_rows="fixed",
            use_container_width=True,
            key="chuva_recente_editor",
            column_config={
                "data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                "precipitacao_mm": st.column_config.NumberColumn("Precipitação (mm)", min_value=0.0, step=0.1, format="%.2f"),
            },
        )

    with aba_futura:
        st.write("Precipitação prevista para cada dia futuro.")
        chuva_futura = st.data_editor(
            criar_tabela_chuva_futura(int(n_dias_futuros)),
            num_rows="dynamic",
            use_container_width=True,
            key=f"chuva_futura_editor_{int(n_dias_futuros)}",
            column_config={
                "data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                "precipitacao_mm": st.column_config.NumberColumn("Precipitação prevista (mm)", min_value=0.0, step=0.1, format="%.2f"),
            },
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ============================================================
# SIMULAÇÃO
# ============================================================
btn_col1, btn_col2 = st.columns([0.20, 0.80])
with btn_col1:
    simular = st.button("Simular vazão", type="primary", use_container_width=True)
with btn_col2:
    st.caption("Revise a chuva recente, a chuva futura e a vazão atual antes de simular.")

if simular:
    try:
        resultado = prever_vazao_futura(
            model=model,
            meta=meta,
            chuva_recente=chuva_recente,
            chuva_futura=chuva_futura,
            q_atual=q_atual,
        )

        st.subheader("Resultado da simulação")
        mostrar_metricas(resultado)

        tab_grafico, tab_tabela = st.tabs(["📈 Gráfico", "📋 Tabela"])
        with tab_grafico:
            grafico_resultados(resultado, meta.get("site_name", sites[site_key]))
        with tab_tabela:
            st.dataframe(formatar_tabela_resultados(resultado), use_container_width=True, hide_index=True)

        csv = resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar resultado em CSV",
            data=csv,
            file_name=f"previsao_vazao_{site_key}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error("Não foi possível executar a simulação. Verifique os dados informados.")
        st.exception(e)

# ============================================================
# REFERÊNCIA E RODAPÉ
# ============================================================
st.markdown(
    f"""
    <div class="reference-box">
        <h4>Referência bibliográfica</h4>
        <p>{BIBLIOGRAPHIC_REFERENCE}</p>
        <p><strong>Link:</strong> <a href="{PROJECT_LINK_URL}" target="_blank">{PROJECT_LINK_LABEL}</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<p class='footer-note'>{METHODOLOGICAL_NOTE}</p>", unsafe_allow_html=True)

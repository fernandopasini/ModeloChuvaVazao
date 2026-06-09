# -*- coding: utf-8 -*-
"""
funcoes_previsao.py
Funções auxiliares para o app Streamlit de previsão de vazão.

Este módulo trabalha com modelos separados por bacia:
- rf_tricolor.pkl
- rf_piray_guazu.pkl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import joblib
import numpy as np
import pandas as pd

DEFAULT_MANIFEST_PATH = "modelos_manifest.json"


def carregar_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de manifesto não encontrado: {path}. "
            "Verifique se modelos_manifest.json está no repositório."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def listar_sites(path: str | Path = DEFAULT_MANIFEST_PATH) -> Dict[str, str]:
    manifest = carregar_manifest(path)
    return {
        site_key: info.get("site_name", site_key)
        for site_key, info in manifest["sites"].items()
    }


def carregar_modelo_por_site(site_key: str, manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> Tuple[Any, Dict[str, Any]]:
    manifest = carregar_manifest(manifest_path)
    if site_key not in manifest["sites"]:
        opcoes = ", ".join(manifest["sites"].keys())
        raise ValueError(f"Site '{site_key}' não reconhecido. Opções: {opcoes}")

    info = manifest["sites"][site_key]
    model_file = Path(info["model_file"])
    if not model_file.exists():
        raise FileNotFoundError(
            f"Arquivo do modelo não encontrado: {model_file}. "
            "Verifique se o .pkl foi enviado ao GitHub junto com o app."
        )

    bundle = joblib.load(model_file)
    return bundle["model"], bundle["meta"]


def _normalizar_coluna_precipitacao(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    candidatos = ["precipitacao_mm", "precipitacao", "chuva", "P", "p"]
    col = next((c for c in candidatos if c in d.columns), None)
    if col is None:
        raise ValueError("A tabela deve conter uma coluna de precipitação, preferencialmente 'precipitacao_mm'.")
    d = d.rename(columns={col: "precipitacao_mm"})
    d["precipitacao_mm"] = pd.to_numeric(d["precipitacao_mm"], errors="coerce").fillna(0.0)
    return d


def _normalizar_coluna_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "data" not in d.columns:
        d["data"] = pd.date_range(pd.Timestamp.today().normalize(), periods=len(d), freq="D")
    d["data"] = pd.to_datetime(d["data"], errors="coerce")
    return d


def validar_entradas_chuva(chuva_recente: pd.DataFrame, chuva_futura: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recente = _normalizar_coluna_data(_normalizar_coluna_precipitacao(chuva_recente))
    futura = _normalizar_coluna_data(_normalizar_coluna_precipitacao(chuva_futura))

    recente = recente.sort_values("data").reset_index(drop=True)
    futura = futura.sort_values("data").reset_index(drop=True)

    p_windows = meta.get("p_windows", [3, 5, 7, 14, 30])
    max_window = int(max(p_windows))
    min_recente = max_window - 1
    if len(recente) < min_recente:
        raise ValueError(
            f"São necessários pelo menos {min_recente} dias de chuva recente para calcular "
            f"os acumulados de até {max_window} dias. Foram informados apenas {len(recente)} dias."
        )
    if len(futura) == 0:
        raise ValueError("Informe pelo menos um dia de precipitação futura.")
    return recente, futura


def prever_vazao_futura(model: Any, meta: Dict[str, Any], chuva_recente: pd.DataFrame, chuva_futura: pd.DataFrame, q_atual: float) -> pd.DataFrame:
    recente, futura = validar_entradas_chuva(chuva_recente, chuva_futura, meta)

    p_col = meta["p_col"]
    q_col = meta["q_col"]
    feature_cols: List[str] = list(meta["features"])
    p_lags = [int(x) for x in meta.get("p_lags", [0, 1, 2, 3, 5, 7, 10, 14])]
    p_windows = [int(x) for x in meta.get("p_windows", [3, 5, 7, 14, 30])]
    q_lags = [int(x) for x in meta.get("q_lags", [1])]

    historico_p = list(recente["precipitacao_mm"].astype(float).values)
    historico_q = [float(q_atual)]

    if max(q_lags) > 1:
        raise ValueError(
            "Este modelo usa mais de uma defasagem de vazão. A interface atual recebe apenas a vazão atual."
        )

    resultados = []
    for _, row in futura.iterrows():
        data_prev = row["data"]
        p_t = float(row["precipitacao_mm"])
        serie_p = historico_p + [p_t]
        feat = {}

        for lag in p_lags:
            idx = len(serie_p) - 1 - lag
            if idx < 0:
                raise ValueError(f"Histórico insuficiente para calcular {p_col}_lag{lag}.")
            feat[f"{p_col}_lag{lag}"] = serie_p[idx]

        for w in p_windows:
            if len(serie_p) < w:
                raise ValueError(f"Histórico insuficiente para calcular {p_col}_sum{w}.")
            feat[f"{p_col}_sum{w}"] = float(np.sum(serie_p[-w:]))

        for ql in q_lags:
            feat[f"{q_col}_lag{ql}"] = historico_q[-ql]

        X_pred = pd.DataFrame([feat]).reindex(columns=feature_cols)
        if X_pred.isna().any(axis=None):
            faltantes = list(X_pred.columns[X_pred.isna().any()])
            raise ValueError(f"Variáveis faltantes para previsão: {faltantes}")

        q_pred = float(model.predict(X_pred)[0])
        resultados.append({
            "data": data_prev,
            "precipitacao_prevista_mm": p_t,
            "vazao_prevista": q_pred,
        })
        historico_p.append(p_t)
        historico_q.append(q_pred)

    return pd.DataFrame(resultados)

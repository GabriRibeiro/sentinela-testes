import numpy as np
import pandas as pd
import pytest

import sentinela as sn

RAZAO_BAR_POR_PSI = 1 / 14.5038


def _linha_bruta(**sobrescritas):
    base = {
        "timestamp": "2026-01-01 00:00:00",
        "id_maquina": "m01",
        "idade_equipamento_meses": 24,
        "id_operador": "op-03",
        "turno": 1,
        "temperatura_c": 60.0,
        "vibracao_rms": 3.0,
        "pressao": 4.0,
        "unidade_pressao": "bar",
        "corrente_a": 20.0,
        "rpm": 1750.0,
        "horas_operacao": 100.0,
        "falha_72h": 0,
    }
    base.update(sobrescritas)
    return base


def test_limpar_normaliza_tipos_e_rotulos():
    bruto = pd.DataFrame([_linha_bruta(id_maquina=" m01 ", id_operador=" op-03 ", unidade_pressao=" BAR ")])
    limpo = sn.preprocessamento.limpar(bruto)

    assert pd.api.types.is_datetime64_any_dtype(limpo["timestamp"])
    assert limpo["id_maquina"].iloc[0] == "M01"
    assert limpo["id_operador"].iloc[0] == "OP-03"
    assert limpo["unidade_pressao"].iloc[0] == "bar"
    assert limpo["turno"].dtype in (np.int64, np.int32)
    assert limpo["idade_equipamento_meses"].dtype in (np.int64, np.int32)


def test_limpar_nao_modifica_o_quadro_de_entrada():
    bruto = pd.DataFrame([_linha_bruta(), _linha_bruta(id_maquina="M02")])
    copia_antes = bruto.copy(deep=True)
    sn.preprocessamento.limpar(bruto)
    pd.testing.assert_frame_equal(bruto, copia_antes)


def test_motor_parado_e_removido():
    bruto = pd.DataFrame([_linha_bruta(rpm=0.0), _linha_bruta(rpm=1750.0)])
    limpo = sn.preprocessamento.limpar(bruto)
    assert len(limpo) == 1
    assert limpo["rpm"].iloc[0] == 1750.0


def test_vibracao_faltante_e_preenchida_mas_fora_da_faixa_de_operacao_bug():
    bruto = pd.DataFrame([_linha_bruta(vibracao_rms=np.nan), _linha_bruta(vibracao_rms=3.0)])
    limpo = sn.preprocessamento.limpar(bruto)

    assert limpo["vibracao_rms"].isna().sum() == 0
    assert limpo["vibracao_rms"].min() >= 1.2, (
        "BUG: vibracao_rms ausente é preenchida com 0.0 (VIBRACAO_PADRAO), um valor fora da faixa "
        f"de operação do sensor (1.2-8.0 mm/s). Valor mínimo obtido após limpar(): "
        f"{limpo['vibracao_rms'].min()}."
    )


@pytest.mark.parametrize("nome_conjunto", ["treino", "teste", "producao"])
def test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug(nome_conjunto):
    bruto = sn.dados.carregar(nome_conjunto)
    limpo = sn.preprocessamento.limpar(bruto)

    por_unidade = limpo.groupby("unidade_pressao")["pressao"].agg(["min", "max", "mean"])
    fora_da_faixa = por_unidade[(por_unidade["min"] < 3.0) | (por_unidade["max"] > 4.5)]

    assert fora_da_faixa.empty, (
        f"BUG: em '{nome_conjunto}', limpar() nunca converte pressão de psi para bar. Faixas por "
        f"unidade após a limpeza:\n{por_unidade}\nA ficha técnica exige 3.0-4.5 bar para todas as "
        "linhas, independente da unidade original."
    )


@pytest.mark.parametrize("nome_conjunto", ["treino", "teste", "producao"])
def test_demais_colunas_respeitam_a_ficha_tecnica_dos_sensores(nome_conjunto):
    bruto = sn.dados.carregar(nome_conjunto)

    assert bruto["temperatura_c"].between(45, 95).all()
    assert bruto["corrente_a"].between(12, 30).all()
    assert bruto["idade_equipamento_meses"].between(6, 180).all()
    rpm_em_operacao = bruto.loc[bruto["rpm"] >= 1.0, "rpm"]
    fracao_fora_da_faixa = (~rpm_em_operacao.between(1650, 1800)).mean()
    assert fracao_fora_da_faixa < 0.005, (
        f"{fracao_fora_da_faixa:.4%} das leituras com motor girando têm rpm fora de 1650-1800 — "
        "tolerável como ruído de borda do sensor, mas confira se não é maior que isso."
    )
    assert bruto["turno"].isin([1, 2, 3]).all()
    assert bruto["id_operador"].str.upper().str.match(r"^OP-(0[1-9]|1[0-2])$").all()

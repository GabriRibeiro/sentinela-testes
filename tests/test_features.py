import numpy as np
import pytest

import sentinela as sn

FEATURES_DE_JANELA_CONTROLE = [
    "temp_max_24h",
    "delta_temp_24h",
    "vib_media_6h",
    "vib_max_24h",
    "corrente_media_6h",
]

FEATURES_SEM_JANELA = [
    "pressao",
    "rpm",
    "idade_equipamento_meses",
    "horas_operacao",
    "operador_senior",
    "turno",
]


def _valor_truncado_vs_completo(limpo, maquina_id, t, feature):
    maquina = (
        limpo[limpo["id_maquina"] == maquina_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    truncado = maquina.iloc[: t + 1]
    x_truncado = sn.features.construir(truncado)
    x_completo = sn.features.construir(maquina)
    return x_truncado.loc[t, feature], x_completo.loc[t, feature]


def test_construir_devolve_exatamente_as_colunas_de_ordem_features(X_teste):
    assert list(X_teste.columns) == list(sn.features.ORDEM_FEATURES)


def test_construir_preserva_a_ordem_das_linhas_de_entrada(limpo_teste):
    embaralhado = limpo_teste.sample(frac=1.0, random_state=1)
    saida = sn.features.construir(embaralhado)
    assert list(saida.index) == list(embaralhado.index)


def test_construir_da_o_mesmo_valor_independente_da_ordem_de_entrada(limpo_teste, X_teste):
    embaralhado = limpo_teste.sample(frac=1.0, random_state=1)
    saida = sn.features.construir(embaralhado).reindex(X_teste.index)
    for coluna in sn.features.ORDEM_FEATURES:
        assert np.allclose(saida[coluna].to_numpy(), X_teste[coluna].to_numpy()), coluna


@pytest.mark.parametrize("feature", FEATURES_DE_JANELA_CONTROLE)
def test_features_de_janela_controle_nao_dependem_do_futuro(limpo_teste, feature):
    maquina_id = limpo_teste["id_maquina"].iloc[0]
    truncado, completo = _valor_truncado_vs_completo(limpo_teste, maquina_id, 20, feature)
    assert truncado == pytest.approx(completo), (
        f"{feature} no instante t=20 não deveria depender de leituras depois de t "
        f"(truncado={truncado}, completo={completo})"
    )


def test_temp_media_6h_depende_do_futuro_bug(limpo_teste):
    maquina_id = limpo_teste["id_maquina"].iloc[0]
    truncado, completo = _valor_truncado_vs_completo(limpo_teste, maquina_id, 20, "temp_media_6h")
    assert truncado == pytest.approx(completo), (
        "BUG: temp_media_6h usa rolling(JANELA_CURTA, center=True), que olha leituras futuras. "
        f"No instante t=20, o valor muda de {truncado} (só com o passado) para {completo} "
        "(com o futuro disponível) — a mesma leitura, dois resultados diferentes."
    )


def test_maquina_risco_nao_deveria_usar_o_proprio_rotulo_do_lote_bug(limpo_teste, X_teste):
    correlacao = np.corrcoef(X_teste["maquina_risco"], limpo_teste["falha_72h"])[0, 1]
    assert abs(correlacao) < 0.15, (
        "BUG: quando o lote tem falha_72h, maquina_risco é a média do próprio rótulo por máquina, "
        f"dentro do mesmo lote sendo avaliado (correlação medida com o rótulo: {correlacao:.3f}). "
        "Isso é a variável usando informação que só existe porque a falha já é conhecida."
    )


@pytest.mark.parametrize("feature", FEATURES_SEM_JANELA)
def test_features_sem_janela_sao_iguais_em_recorte_e_no_lote_inteiro(limpo_teste, X_teste, feature):
    recorte = limpo_teste.sample(n=50, random_state=3)
    x_recorte = sn.features.construir(recorte)
    esperado = X_teste.loc[recorte.index, feature].to_numpy()
    obtido = x_recorte[feature].to_numpy()
    assert np.allclose(obtido, esperado), feature

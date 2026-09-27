import numpy as np
import pytest
from scipy.stats import chi2, ks_2samp

import sentinela as sn


@pytest.fixture(scope="module")
def saida_v1_teste(bruto_teste):
    return sn.pipeline.executar(bruto_teste, versao="v1")


@pytest.fixture(scope="module")
def saida_v2_teste(bruto_teste):
    return sn.pipeline.executar(bruto_teste, versao="v2")


def test_prevalencia_e_baseline_trivial_mostram_que_acuracia_engana(saida_v1_teste):
    y = saida_v1_teste["falha_72h"].to_numpy()
    prevalencia = y.mean()
    acuracia_baseline = (y == 0).mean()

    assert prevalencia < 0.25
    assert acuracia_baseline > 0.7, (
        f"um modelo que nunca prevê falha acerta {100*acuracia_baseline:.1f}% das vezes, com "
        f"prevalência real de {100*prevalencia:.1f}% — acurácia sozinha não distingue esse modelo "
        "inútil de um modelo que de fato detecta falha; recall/precisão/F1 respondem melhor."
    )


def test_v2_tem_mais_acuracia_mas_recall_pior_que_v1(saida_v1_teste, saida_v2_teste):
    y = saida_v1_teste["falha_72h"].to_numpy()
    m1 = sn.avaliacao.metricas(y, saida_v1_teste["predicao"].to_numpy())
    m2 = sn.avaliacao.metricas(y, saida_v2_teste["predicao"].to_numpy())

    assert m2["acuracia"] > m1["acuracia"]
    assert m2["recall"] < m1["recall"], (
        f"v2 acerta mais no geral (acurácia {m2['acuracia']:.3f} vs {m1['acuracia']:.3f}) trocando "
        f"recall (v2={m2['recall']:.3f} vs v1={m1['recall']:.3f}) — cada falha perdida é um motor "
        "que pode queimar sem aviso; a promoção da v2 não é obviamente boa."
    )


def test_bootstrap_ic_confirma_que_a_perda_de_recall_da_v2_nao_e_ruido(saida_v1_teste, saida_v2_teste):
    y = saida_v1_teste["falha_72h"].to_numpy()
    p1 = saida_v1_teste["predicao"].to_numpy()
    p2 = saida_v2_teste["predicao"].to_numpy()
    n = len(y)
    rng = np.random.default_rng(42)

    def _acc_recall(y_am, p_am):
        vp = np.sum((y_am == 1) & (p_am == 1))
        vn = np.sum((y_am == 0) & (p_am == 0))
        fn = np.sum((y_am == 1) & (p_am == 0))
        acc = (vp + vn) / len(y_am)
        recall = vp / (vp + fn) if (vp + fn) else 0.0
        return acc, recall

    b = 2000
    deltas_acc = np.empty(b)
    deltas_recall = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        acc1, rec1 = _acc_recall(y[idx], p1[idx])
        acc2, rec2 = _acc_recall(y[idx], p2[idx])
        deltas_acc[i] = acc2 - acc1
        deltas_recall[i] = rec2 - rec1

    ic_acc = np.percentile(deltas_acc, [2.5, 97.5])
    ic_recall = np.percentile(deltas_recall, [2.5, 97.5])

    assert ic_acc[0] > 0, f"IC 95% da diferença de acurácia (v2-v1) é {ic_acc}, não deveria cruzar zero"
    assert ic_recall[1] < 0, (
        f"IC 95% da diferença de recall (v2-v1) é {ic_recall} — inteiramente negativo: a perda de "
        "recall da v2 é sistemática, não ruído de amostragem."
    )


def test_mcnemar_confirma_que_v1_e_v2_discordam_de_forma_nao_aleatoria(saida_v1_teste, saida_v2_teste):
    p1 = saida_v1_teste["predicao"].to_numpy()
    p2 = saida_v2_teste["predicao"].to_numpy()
    b01 = int(np.sum((p1 == 0) & (p2 == 1)))
    b10 = int(np.sum((p1 == 1) & (p2 == 0)))
    estatistica = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_valor = 1 - chi2.cdf(estatistica, df=1)

    assert p_valor < 0.01, (
        f"McNemar: v1=0,v2=1 -> {b01} casos | v1=1,v2=0 -> {b10} casos, p={p_valor:.4g} — as "
        "decisões de v1 e v2 divergem de forma sistemática, não por acaso."
    )


def test_limiar_0_5_nao_maximiza_f1_bug(X_teste, limpo_teste, modelo_v1):
    y = limpo_teste["falha_72h"].to_numpy()
    proba = modelo_v1.prever_proba(X_teste)

    def _f1(limiar):
        p = (proba >= limiar).astype(int)
        return sn.avaliacao.metricas(y, p)["f1"]

    f1_padrao = _f1(0.5)
    candidatos = {limiar: _f1(limiar) for limiar in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    melhor_limiar = max(candidatos, key=candidatos.get)

    assert candidatos[melhor_limiar] <= f1_padrao + 1e-9, (
        f"BUG (decisão de projeto, não de código): limiar 0.5 dá F1={f1_padrao:.3f}; limiar "
        f"{melhor_limiar} dá F1={candidatos[melhor_limiar]:.3f} — o limiar em produção não é o "
        "que maximiza F1 no conjunto de teste."
    )


def test_probabilidades_da_v1_nao_sao_calibradas_na_faixa_baixa_e_media_bug(X_teste, limpo_teste, modelo_v1):
    y = limpo_teste["falha_72h"].to_numpy()
    proba = modelo_v1.prever_proba(X_teste)
    bins = np.linspace(0, 1, 11)
    indices = np.clip(np.digitize(proba, bins) - 1, 0, 9)

    maior_erro = 0.0
    faixa_do_erro = None
    for faixa in range(4):
        mascara = indices == faixa
        if mascara.sum() < 30:
            continue
        media_prevista = proba[mascara].mean()
        freq_observada = y[mascara].mean()
        erro = abs(media_prevista - freq_observada)
        if erro > maior_erro:
            maior_erro = erro
            faixa_do_erro = (bins[faixa], bins[faixa + 1], media_prevista, freq_observada)

    assert maior_erro < 0.1, (
        f"BUG: na faixa [{faixa_do_erro[0]:.1f}-{faixa_do_erro[1]:.1f}], a v1 prevê probabilidade "
        f"média {faixa_do_erro[2]:.3f} mas a frequência real observada é {faixa_do_erro[3]:.3f} "
        f"(erro de {maior_erro:.3f}) — a reponderação de classe no treino provavelmente distorce a "
        "probabilidade, que deixa de ser uma probabilidade de verdade nessa faixa."
    )


@pytest.mark.parametrize(
    "coluna", ["temperatura_c", "vibracao_rms", "pressao", "corrente_a", "rpm"]
)
def test_distribuicao_teste_e_producao_sem_drift_pratico_relevante(bruto_teste, bruto_producao, coluna):
    a = bruto_teste[coluna].dropna().to_numpy()
    b = bruto_producao[coluna].dropna().to_numpy()
    estatistica, p_valor = ks_2samp(a, b)

    assert estatistica < 0.15, (
        f"{coluna}: KS={estatistica:.4f} (p={p_valor:.4g}) — diferença de distribuição grande "
        "demais para ser só ruído de amostragem, mesmo com médias parecidas entre teste e produção."
    )

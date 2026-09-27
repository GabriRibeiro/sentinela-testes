import numpy as np

import sentinela as sn


def test_perturbacao_menor_que_o_ruido_do_sensor_nao_deveria_mudar_a_decisao_bug(bruto_teste, modelo_v1):
    limpo = sn.preprocessamento.limpar(bruto_teste)
    X_base = sn.features.construir(limpo)
    p_base = modelo_v1.prever(X_base)

    rng = np.random.default_rng(7)
    perturbado = bruto_teste.copy()
    perturbado["temperatura_c"] = perturbado["temperatura_c"] + rng.uniform(-0.5, 0.5, size=len(perturbado))
    limpo_perturbado = sn.preprocessamento.limpar(perturbado)
    X_perturbado = sn.features.construir(limpo_perturbado)
    p_perturbado = modelo_v1.prever(X_perturbado)

    taxa = (p_base != p_perturbado).mean()
    assert taxa == 0, (
        "BUG: perturbar a temperatura em até ±0.5°C — metade do ruído declarado do sensor "
        f"(±1.0°C, README) — muda a decisão em {100*taxa:.2f}% das leituras. Uma ordem de "
        "manutenção não devia depender de uma variação menor que a precisão do próprio instrumento."
    )


def test_maquina_risco_e_a_dependencia_mais_forte_e_e_causalmente_ilegitima_bug(X_teste, modelo_v1):
    base = X_teste.copy()
    p_base = modelo_v1.prever_proba(base)

    impacto = {}
    for coluna in sn.features.ORDEM_FEATURES:
        alterado = base.copy()
        desvio = alterado[coluna].std() or 1.0
        alterado[coluna] = alterado[coluna] + desvio
        p_alterado = modelo_v1.prever_proba(alterado)
        impacto[coluna] = float(np.abs(p_alterado - p_base).mean())

    mais_influente = max(impacto, key=impacto.get)
    assert mais_influente not in {"maquina_risco"}, (
        f"BUG: a feature que mais move a probabilidade quando deslocada em 1 desvio-padrão é "
        f"'{mais_influente}' (impacto médio {impacto[mais_influente]:.4f}, o dobro da segunda "
        f"colocada). Ela não descreve o estado físico atual do motor — é um resumo do histórico "
        "de falhas da própria máquina, incluindo, no lote rotulado, o rótulo que está sendo previsto."
    )


def test_contrafactual_turno_nao_deveria_mudar_a_decisao(X_teste, modelo_v1):
    amostra = X_teste.sample(n=300, random_state=11)
    p_base = modelo_v1.prever(amostra)

    mudou = np.zeros(len(amostra), dtype=bool)
    for novo_turno in (1.0, 2.0, 3.0):
        variante = amostra.copy()
        variante["turno"] = novo_turno
        p_variante = modelo_v1.prever(variante)
        mudou |= p_variante != p_base

    taxa = mudou.mean()
    assert taxa < 0.05, (
        f"trocar só o turno (mantendo o resto do estado físico do motor igual) muda a decisão em "
        f"{100*taxa:.1f}% dos casos — turno não deveria, isoladamente, decidir uma ordem de "
        "manutenção."
    )


def test_lote_com_um_motor_so_nao_quebra(bruto_teste):
    maquina_id = bruto_teste["id_maquina"].iloc[0]
    um_motor = bruto_teste[bruto_teste["id_maquina"] == maquina_id]
    saida = sn.pipeline.executar(um_motor, versao="v1")
    assert len(saida) > 0


def test_leitura_extrema_fora_da_ficha_tecnica_nao_quebra_a_predicao(bruto_teste):
    extremo = bruto_teste.iloc[[0]].copy()
    extremo["temperatura_c"] = 500.0
    extremo["vibracao_rms"] = -10.0
    saida = sn.pipeline.executar(extremo, versao="v1")
    assert saida["probabilidade"].between(0, 1).all()

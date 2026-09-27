import random

import numpy as np

import sentinela as sn


def test_prever_registro_concorda_com_prever_proba_quando_a_ordem_esta_certa(X_teste, modelo_v1):
    linha = X_teste.iloc[0]
    registro = {nome: float(linha[nome]) for nome in sn.features.ORDEM_FEATURES}

    predicao_registro = modelo_v1.prever_registro(registro)
    predicao_matriz = int(modelo_v1.prever(np.array([list(registro.values())]))[0])

    assert predicao_registro == predicao_matriz


def test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug(X_teste, modelo_v1):
    n = 200
    divergencias = 0
    for i in range(n):
        linha = X_teste.iloc[i]
        registro = {nome: float(linha[nome]) for nome in sn.features.ORDEM_FEATURES}

        chaves_embaralhadas = list(sn.features.ORDEM_FEATURES)
        random.seed(i)
        random.shuffle(chaves_embaralhadas)
        registro_embaralhado = {chave: registro[chave] for chave in chaves_embaralhadas}

        original = modelo_v1.prever_registro(registro)
        embaralhado = modelo_v1.prever_registro(registro_embaralhado)
        if original != embaralhado:
            divergencias += 1

    taxa = divergencias / n
    assert divergencias == 0, (
        "BUG: prever_registro monta o vetor de entrada a partir de dict.values(), na ordem de "
        "inserção do dicionário, sem validar as chaves contra ordem_features. Reembaralhar as "
        f"chaves do mesmo registro (mesmos valores) muda a decisão em {divergencias}/{n} casos "
        f"({100*taxa:.1f}%). É o caminho usado pelo painel da sala de controle."
    )


def test_pipeline_executar_fim_a_fim(bruto_teste):
    saida = sn.pipeline.executar(bruto_teste.head(500), versao="v1")

    assert list(saida.columns[:4]) == ["id_maquina", "timestamp", "probabilidade", "predicao"]
    assert saida["probabilidade"].between(0, 1).all()
    assert set(saida["predicao"].unique()) <= {0, 1}


def test_artefatos_e_dados_batem_com_o_manifest():
    problemas = sn.artefatos.verificar_manifest()
    assert problemas == [], f"divergências encontradas: {problemas}"


def test_lote_com_um_motor_so(bruto_teste):
    maquina_id = bruto_teste["id_maquina"].iloc[0]
    um_motor = bruto_teste[bruto_teste["id_maquina"] == maquina_id]
    saida = sn.pipeline.executar(um_motor, versao="v1")
    assert len(saida) > 0


def test_lote_com_uma_linha_so(bruto_teste):
    uma_linha = bruto_teste.iloc[[0]]
    saida = sn.pipeline.executar(uma_linha, versao="v1")
    assert len(saida) == 1


def test_leitura_fora_da_faixa_da_ficha_tecnica_nao_quebra_a_predicao(bruto_teste):
    extremo = bruto_teste.iloc[[0]].copy()
    extremo["temperatura_c"] = 500.0
    extremo["vibracao_rms"] = -10.0
    saida = sn.pipeline.executar(extremo, versao="v1")
    assert saida["probabilidade"].between(0, 1).all()

import pytest

import sentinela as sn


@pytest.fixture(scope="session")
def bruto_treino():
    return sn.dados.carregar("treino")


@pytest.fixture(scope="session")
def bruto_teste():
    return sn.dados.carregar("teste")


@pytest.fixture(scope="session")
def bruto_producao():
    return sn.dados.carregar("producao")


@pytest.fixture(scope="session")
def limpo_teste(bruto_teste):
    return sn.preprocessamento.limpar(bruto_teste)


@pytest.fixture(scope="session")
def X_teste(limpo_teste):
    return sn.features.construir(limpo_teste)


@pytest.fixture(scope="session")
def modelo_v1():
    return sn.modelo.carregar("v1")


@pytest.fixture(scope="session")
def modelo_v2():
    return sn.modelo.carregar("v2")

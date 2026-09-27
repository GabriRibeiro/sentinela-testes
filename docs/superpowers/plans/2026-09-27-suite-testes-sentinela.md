# Suíte de testes do Sentinela — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir e comprovar, com `pytest` rodando de verdade, uma suíte de testes externa ao pacote `sentinela` que cobre os três blocos obrigatórios da Trilha 1 (unitário, estatístico, adversarial) e documenta pelo menos um defeito real com evidência medida.

**Architecture:** Repositório Python independente (`sentinela-testes/`) que instala `sentinela` como dependência editável a partir de um clone irmão (`../sentinela`, nunca editado) e importa o pacote só pela interface pública (`sn.dados`, `sn.preprocessamento`, `sn.features`, `sn.modelo`, `sn.pipeline`, `sn.avaliacao`, `sn.artefatos`). Fixtures de sessão em `conftest.py` carregam os três conjuntos de dados e os dois modelos uma única vez; cada arquivo de teste ataca um bloco da rubrica.

**Tech Stack:** Python 3.12, pytest, pandas, numpy, scipy (bootstrap manual + `scipy.stats.chi2`/`ks_2samp`).

## Global Constraints

- Não editar nenhum arquivo dentro do clone `sentinela` (`../sentinela/src/sentinela/**`). Toda correção vira `.patch` em `correcoes/`, nunca um edit direto.
- A suíte final deve rodar com `pytest -v` sem erro de coleta. Testes que documentam um defeito devem falhar de verdade (sem `xfail`/`skip`) — a falha é a evidência.
- Nenhum teste pode depender de ordem de execução entre arquivos; toda semente aleatória usada em um teste deve ser fixa (reprodutível).
- Nomes de função, variável e mensagens de asserção em português, sem comentário redundante (o nome e a mensagem já explicam).

---

## Arquivos que este plano cria

```
sentinela-testes/
  requirements.txt
  pytest.ini
  .gitignore
  tests/
    conftest.py
    test_preprocessamento.py
    test_features.py
    test_pipeline.py
    test_estatistico.py
    test_adversarial.py
  correcoes/
    README.md
    01-pressao-psi-bar.patch
    02-temp-media-6h-look-ahead.patch
    03-prever-registro-ordem-chaves.patch
  README.md
  relatorio.md
```

E, como dependência irmã (fora deste repositório, mas necessária para instalar e rodar):

```
../sentinela/          # clone oficial, não editado, instalado via pip install -e .
```

---

### Task 0: Ambiente do projeto (venv, dependência `sentinela`, fixtures)

**Files:**
- Create: `sentinela-testes/requirements.txt`
- Create: `sentinela-testes/pytest.ini`
- Create: `sentinela-testes/.gitignore`
- Create: `sentinela-testes/tests/conftest.py`

**Interfaces:**
- Produces: fixtures de sessão `bruto_treino`, `bruto_teste`, `bruto_producao`, `limpo_teste`, `X_teste`, `modelo_v1`, `modelo_v2`, usadas por todas as tasks seguintes.

- [ ] **Step 1: Clonar o `sentinela` como dependência irmã (se ainda não existir)**

```bash
cd "C:/Users/User/Documents/repositorio"
test -d sentinela || git clone https://github.com/felipehp/sentinela-nortemec.git sentinela
```

- [ ] **Step 2: Criar o ambiente virtual do projeto de testes**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
"/c/Users/User/AppData/Local/Programs/Python/Python312/python.exe" -m venv .venv
```

- [ ] **Step 3: Criar `requirements.txt`**

```
pandas
numpy
scipy
pytest
```

- [ ] **Step 4: Instalar dependências e o pacote `sentinela` (editável, a partir do clone irmão)**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -e ../sentinela
```

- [ ] **Step 5: Verificar a instalação**

```bash
./.venv/Scripts/python.exe -c "import sentinela as sn; print(sn.dados.carregar('teste').shape)"
```

Expected: `(4200, 13)`

- [ ] **Step 6: Criar `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -v
```

- [ ] **Step 7: Criar `.gitignore`**

```
.venv/
__pycache__/
.pytest_cache/
*.pyc
```

- [ ] **Step 8: Criar `tests/conftest.py`**

```python
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
```

- [ ] **Step 9: Rodar a coleta de testes (ainda vazia) para confirmar que o ambiente está sadio**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest --collect-only
```

Expected: `no tests ran` sem erro de importação.

- [ ] **Step 10: Commit**

```bash
git add requirements.txt pytest.ini .gitignore tests/conftest.py
git commit -m "Configura ambiente do projeto e fixtures de dados/modelos"
```

---

### Task 1: `test_preprocessamento.py` — Bloco A (tipos, faltantes, faixas/unidades)

**Files:**
- Create: `sentinela-testes/tests/test_preprocessamento.py`

**Interfaces:**
- Consumes: fixtures `bruto_teste`, `bruto_treino`, `bruto_producao` (Task 0); `sn.preprocessamento.limpar(df) -> pd.DataFrame`.

- [ ] **Step 1: Escrever o arquivo de teste completo**

```python
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
    assert limpo["turno"].dtype == np.int64 or limpo["turno"].dtype == np.int32
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
    assert bruto.loc[bruto["rpm"] >= 1.0, "rpm"].between(1650, 1800).all()
    assert bruto["turno"].isin([1, 2, 3]).all()
    assert bruto["id_operador"].str.upper().str.match(r"^OP-(0[1-9]|1[0-2])$").all()
```

- [ ] **Step 2: Rodar e observar o resultado**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest tests/test_preprocessamento.py -v
```

Expected: `test_limpar_normaliza_tipos_e_rotulos`, `test_limpar_nao_modifica_o_quadro_de_entrada`,
`test_motor_parado_e_removido` e `test_demais_colunas_respeitam_a_ficha_tecnica_dos_sensores` **PASSAM**.
`test_vibracao_faltante_e_preenchida_mas_fora_da_faixa_de_operacao_bug` e
`test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug` (as 3 parametrizações) **FALHAM**
— essa falha é a evidência do defeito. Se algum teste que deveria passar falhar, investigue antes de
seguir (releia `preprocessamento.py`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_preprocessamento.py
git commit -m "Adiciona testes unitarios de preprocessamento (Bloco A)"
```

---

### Task 2: `test_features.py` — Bloco A (ordem, determinismo, vazamento temporal, vazamento de rótulo)

**Files:**
- Create: `sentinela-testes/tests/test_features.py`

**Interfaces:**
- Consumes: fixtures `limpo_teste`, `X_teste` (Task 0); `sn.features.construir(df) -> pd.DataFrame`;
  `sn.features.ORDEM_FEATURES: tuple[str, ...]`.

- [ ] **Step 1: Escrever o arquivo de teste completo**

```python
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
```

- [ ] **Step 2: Rodar e observar o resultado**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest tests/test_features.py -v
```

Expected: tudo passa **exceto** `test_temp_media_6h_depende_do_futuro_bug` e
`test_maquina_risco_nao_deveria_usar_o_proprio_rotulo_do_lote_bug`, que falham — evidência dos
defeitos 2 e 3. Se `test_features_de_janela_controle_nao_dependem_do_futuro` falhar para alguma das 5
features de controle, pare e reanalise (não é o comportamento esperado).

- [ ] **Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "Adiciona testes de features: ordem, determinismo e vazamentos (Bloco A)"
```

---

### Task 3: `test_pipeline.py` — Bloco A (contrato de `prever_registro`, integração, reprodutibilidade)

**Files:**
- Create: `sentinela-testes/tests/test_pipeline.py`

**Interfaces:**
- Consumes: fixtures `X_teste`, `modelo_v1`, `bruto_teste` (Task 0); `Modelo.prever_registro(dict) -> int`;
  `Modelo.prever(X, limiar) -> np.ndarray`; `sn.pipeline.executar(df, versao, limiar) -> pd.DataFrame`;
  `sn.artefatos.verificar_manifest() -> list[str]`.

- [ ] **Step 1: Escrever o arquivo de teste completo**

```python
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
```

- [ ] **Step 2: Rodar e observar o resultado**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v
```

Expected: tudo passa **exceto** `test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug`,
que falha (evidência do defeito 4). Divergência esperada em torno de 30-35% em 200 casos — se vier 0,
confira se os valores usados realmente diferem por posição (releia `Modelo.prever_registro` em
`modelo.py`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "Adiciona testes de contrato do pipeline e prever_registro (Bloco A)"
```

---

### Task 4: `test_estatistico.py` — Bloco B (métrica adequada, v1×v2, limiar, calibração, drift)

**Files:**
- Create: `sentinela-testes/tests/test_estatistico.py`

**Interfaces:**
- Consumes: fixtures `bruto_teste`, `bruto_producao`, `limpo_teste`, `X_teste`, `modelo_v1` (Task 0);
  `sn.pipeline.executar`; `sn.avaliacao.metricas`.

- [ ] **Step 1: Escrever o arquivo de teste completo**

```python
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
```

- [ ] **Step 2: Rodar e observar o resultado**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest tests/test_estatistico.py -v
```

Expected: `test_prevalencia_...`, `test_v2_tem_mais_acuracia...`, `test_bootstrap_ic...`,
`test_mcnemar...` e as 5 parametrizações de `test_distribuicao_teste_e_producao...` **PASSAM**.
`test_limiar_0_5_nao_maximiza_f1_bug` e `test_probabilidades_da_v1_nao_sao_calibradas...` **FALHAM**
— evidência dos achados de limiar e calibração.

- [ ] **Step 3: Commit**

```bash
git add tests/test_estatistico.py
git commit -m "Adiciona testes estatisticos: metricas, v1 vs v2, limiar, calibracao, drift (Bloco B)"
```

---

### Task 5: `test_adversarial.py` — Bloco C (ruído, importância, contrafactual, casos-limite)

**Files:**
- Create: `sentinela-testes/tests/test_adversarial.py`

**Interfaces:**
- Consumes: fixtures `bruto_teste`, `X_teste`, `modelo_v1` (Task 0).

- [ ] **Step 1: Escrever o arquivo de teste completo**

```python
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
```

- [ ] **Step 2: Rodar e observar o resultado**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest tests/test_adversarial.py -v
```

Expected: `test_contrafactual_turno...`, `test_lote_com_um_motor_so...` e
`test_leitura_extrema...` **PASSAM**. `test_perturbacao_menor_que_o_ruido_do_sensor...` e
`test_maquina_risco_e_a_dependencia_mais_forte...` **FALHAM** — evidência dos achados de robustez e
legitimidade causal.

- [ ] **Step 3: Commit**

```bash
git add tests/test_adversarial.py
git commit -m "Adiciona testes adversariais: ruido, importancia, contrafactual (Bloco C)"
```

---

### Task 6: Rodar a suíte completa e escrever `README.md` com o log real

**Files:**
- Create: `sentinela-testes/README.md`

**Interfaces:**
- Consumes: nada de código; usa a suíte completa das Tasks 0-5.

- [ ] **Step 1: Rodar a suíte inteira e salvar a saída**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest -v > docs/log_execucao.txt 2>&1
cat docs/log_execucao.txt
```

Expected: um misto de PASSED e FAILED. Confira que os FAILED são exatamente os 6 testes marcados
"_bug" nas tasks anteriores (pressão, temp_media_6h, maquina_risco, prever_registro, limiar,
calibração, ruído sub-sensor, maquina_risco-importância — 8 no total). Qualquer FAILED fora dessa
lista é um problema a investigar antes de seguir; qualquer um desses 8 que vier PASSED também exige
investigação (o teste pode estar formulado errado).

- [ ] **Step 2: Escrever `README.md`, com o conteúdo de `docs/log_execucao.txt` colado na seção final**

```markdown
# Suíte de testes — Sentinela (Trilha 1)

Suíte de testes externa ao pacote `sentinela` (https://github.com/felipehp/sentinela-nortemec),
cobrindo os três blocos obrigatórios da Trilha 1: unitário, estatístico e adversarial. O pacote
nunca é editado — todo defeito encontrado é documentado com um teste que falha.

## Como rodar

Precisa de Python 3.10+.

```bash
git clone https://github.com/felipehp/sentinela-nortemec.git ../sentinela
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -e ../sentinela
./.venv/Scripts/python.exe -m pytest -v
```

## O que esperar

A suíte tem testes que devem passar e testes que devem falhar de propósito — cada falha documenta
um defeito real do sistema, com o número medido na própria mensagem de asserção. Ver `relatorio.md`
para a análise de cada um.

## Log de execução real

```
<COLAR AQUI O CONTEÚDO INTEGRAL DE docs/log_execucao.txt>
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/log_execucao.txt
git commit -m "Documenta como rodar a suite e cola o log real de execucao"
```

---

### Task 7: Correções opcionais (patches para os defeitos 1, 2 e 4)

**Files:**
- Create: `sentinela-testes/correcoes/README.md`
- Create: `sentinela-testes/correcoes/01-pressao-psi-bar.patch`
- Create: `sentinela-testes/correcoes/02-temp-media-6h-look-ahead.patch`
- Create: `sentinela-testes/correcoes/03-prever-registro-ordem-chaves.patch`
- Não modifica: nenhum arquivo do clone `../sentinela` permanentemente (a branch de correção é
  descartada ao final deste task).

**Interfaces:**
- Consumes: o clone `../sentinela` (Task 0) e os testes que documentam o bug em `test_preprocessamento.py`, `test_features.py`, `test_pipeline.py`.

- [ ] **Step 1: Criar uma branch de correção no clone irmão**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela"
git checkout -b correcao/defeitos-confirmados
```

- [ ] **Step 2: Corrigir o defeito 1 (pressão psi/bar) em `src/sentinela/preprocessamento.py`**

Adicionar, depois da linha que normaliza `unidade_pressao` e antes do filtro de `rpm`:

```python
    psi_para_bar = 1 / 14.5038
    em_psi = limpo["unidade_pressao"] == "psi"
    limpo.loc[em_psi, "pressao"] = limpo.loc[em_psi, "pressao"] * psi_para_bar
    limpo["unidade_pressao"] = "bar"
```

- [ ] **Step 3: Corrigir o defeito 2 (look-ahead) em `src/sentinela/features.py`**

Trocar:

```python
    saida["temp_media_6h"] = temperatura.transform(
        lambda s: s.rolling(JANELA_CURTA, center=True, min_periods=1).mean()
    )
```

por:

```python
    saida["temp_media_6h"] = temperatura.transform(
        lambda s: s.rolling(JANELA_CURTA, min_periods=1).mean()
    )
```

- [ ] **Step 4: Corrigir o defeito 4 (ordem do dicionário) em `src/sentinela/modelo.py`**

Trocar o corpo de `prever_registro`:

```python
    def prever_registro(self, registro: dict, limiar: float = LIMIAR_PADRAO) -> int:
        faltando = set(self.ordem_features) - set(registro)
        if faltando:
            raise ValueError(f"faltam features no registro: {sorted(faltando)}")
        vetor = np.array(
            [float(registro[nome]) for nome in self.ordem_features], dtype=np.float64
        )
        return int(self.prever(vetor.reshape(1, -1), limiar=limiar)[0])
```

- [ ] **Step 5: Commitar a branch de correção e gerar os patches**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela"
git add -A
git commit -m "Corrige pressao psi/bar, look-ahead em temp_media_6h e ordem em prever_registro"

git diff main -- src/sentinela/preprocessamento.py > ../sentinela-testes/correcoes/01-pressao-psi-bar.patch
git diff main -- src/sentinela/features.py > ../sentinela-testes/correcoes/02-temp-media-6h-look-ahead.patch
git diff main -- src/sentinela/modelo.py > ../sentinela-testes/correcoes/03-prever-registro-ordem-chaves.patch
```

- [ ] **Step 6: Reinstalar o pacote corrigido no venv de testes e confirmar que os testes correspondentes passam a verde**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pip install -e ../sentinela --force-reinstall --no-deps

./.venv/Scripts/python.exe -m pytest \
  tests/test_preprocessamento.py::test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug \
  tests/test_features.py::test_temp_media_6h_depende_do_futuro_bug \
  tests/test_pipeline.py::test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug \
  -v > docs/log_correcoes.txt 2>&1
cat docs/log_correcoes.txt
```

Expected: os 3 testes (agora `test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug` só
para os 3 conjuntos) **PASSAM** contra o código corrigido.

- [ ] **Step 7: Desfazer a instalação corrigida e voltar ao pacote original (para não invalidar as Tasks 1-6)**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela"
git checkout main

cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pip install -e ../sentinela --force-reinstall --no-deps
./.venv/Scripts/python.exe -m pytest tests/test_features.py::test_temp_media_6h_depende_do_futuro_bug -v
```

Expected: volta a **FALHAR** — confirma que o venv está de novo apontando para o pacote original, sem
correção, como o resto da suíte espera.

- [ ] **Step 8: Escrever `correcoes/README.md`**

```markdown
# Correções (bônus, opcionais)

Cada patch corrige um defeito confirmado pela suíte principal, sem tocar no pacote usado pelo
restante dos testes — o patch é gerado a partir de uma branch separada no clone `../sentinela` e
descartado depois de comprovado.

## 01 — pressão em psi não convertida para bar

Corrige `preprocessamento.limpar`: converte toda leitura marcada `unidade_pressao == "psi"` para
bar (1 bar = 14,5038 psi) e normaliza a coluna `unidade_pressao` para `"bar"`.

Teste que passa a verde: `tests/test_preprocessamento.py::test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug`

## 02 — `temp_media_6h` olha para o futuro

Corrige `features.construir`: remove `center=True` do `rolling` de `temp_media_6h`, que passa a
depender só de leituras até o instante atual, como as demais features de janela.

Teste que passa a verde: `tests/test_features.py::test_temp_media_6h_depende_do_futuro_bug`

## 03 — `prever_registro` ignora a ordem das chaves do dicionário

Corrige `Modelo.prever_registro`: monta o vetor de entrada indexando o dicionário pelas chaves de
`ordem_features`, em vez de confiar na ordem de inserção de `dict.values()`, e valida que todas as
chaves esperadas estão presentes.

Teste que passa a verde: `tests/test_pipeline.py::test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug`

## Sem correção: vazamento em `maquina_risco`

O defeito 3 (a feature usa o próprio rótulo do lote quando ele existe) não tem patch aqui: corrigi-lo
de verdade significa mudar a definição da feature e retreinar os dois modelos, o que está fora do
escopo de um time de testes. Fica documentado em `relatorio.md`.

## Como reproduzir a verificação

```bash
cd ../sentinela
git log --oneline -3   # confirma que main está limpo, sem a branch de correção
```

O log da comprovação (testes passando contra o código corrigido) está em `../docs/log_correcoes.txt`.
```

- [ ] **Step 9: Commit no repositório de testes**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
git add correcoes/ docs/log_correcoes.txt
git commit -m "Adiciona patches de correcao para os defeitos 1, 2 e 4, com evidencia de que passam"
```

---

### Task 8: `relatorio.md` — achados, evidência, causa raiz, impacto

**Files:**
- Create: `sentinela-testes/relatorio.md`

**Interfaces:**
- Consumes: os números medidos nas Tasks 1-5 (já confirmados: correlação 0,50; taxa 32,5%; diferenças
  de acurácia/recall com IC; maior erro de calibração ~0,34; impacto de `maquina_risco` ~0,11).

- [ ] **Step 1: Escrever o relatório completo**

```markdown
# Relatório — suíte de testes do Sentinela

Autor: individual. Sistema sob teste: https://github.com/felipehp/sentinela-nortemec (não editado).

## Como a suíte foi pensada

Bloco A ataca `preprocessamento.limpar` e `features.construir` isoladamente, validando o contrato de
dados da ficha técnica e a garantia de que uma feature no instante *t* só pode depender de leituras
até *t*. Bloco B compara v1 e v2 com incerteza, em vez de comparar dois números soltos, e questiona o
limiar e a calibração herdados do time de desenvolvimento. Bloco C testa se a decisão resiste a
perturbações menores que o ruído do próprio sensor e se alguma dependência do modelo é causalmente
ilegítima.

## Defeitos confirmados

### 1. Pressão em unidade mista (psi/bar) nunca convertida

`preprocessamento.limpar` normaliza `unidade_pressao` para minúsculo, mas nunca converte o valor de
`pressao` quando a unidade é psi. Cerca de 32% das linhas dos três conjuntos chegam em psi
(faixa ~48-65) e ficam misturadas, sem conversão, com as linhas em bar (faixa ~3-4,5) — a mesma
coluna, duas escalas.

- **Teste que falha:** `tests/test_preprocessamento.py::test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug`
- **Evidência medida:** ver `docs/log_execucao.txt` — a mensagem de falha traz min/max/média de
  `pressao` por `unidade_pressao`, com o grupo `psi` fora da faixa 3,0-4,5 em todos os três conjuntos.
- **Causa raiz:** falta a linha de conversão (1 bar = 14,5038 psi) em `limpar()`; a normalização de
  string (`.str.lower()`) engana quem lê o código, pois parece que a unidade "foi tratada".
- **Impacto na planta:** a feature `pressao` alimenta o modelo com dois significados físicos
  diferentes para o mesmo número — para ~32% dos motores, a "pressão" que o modelo vê pode estar até
  15x maior que a pressão real do motor. Qualquer relação aprendida entre pressão e falha está
  contaminada por essa mistura de escalas.

### 2. `temp_media_6h` depende de leituras futuras (look-ahead)

`features.construir` calcula `temp_media_6h` com `rolling(JANELA_CURTA, center=True, min_periods=1)`.
`center=True` usa leituras depois do instante *t* para compor o valor em *t*.

- **Teste que falha:** `tests/test_features.py::test_temp_media_6h_depende_do_futuro_bug`
- **Evidência medida:** truncar a série de uma máquina no instante t=20 muda o valor de
  `temp_media_6h` em relação à série completa (dois valores diferentes para o mesmo instante,
  registrados no log de execução).
- **Causa raiz:** `center=True` no `rolling`; as demais janelas (`temp_max_24h`, `vib_media_6h` etc.)
  não usam esse parâmetro e passam no mesmo teste de controle.
- **Impacto na planta:** o modelo foi treinado e avaliado com uma feature que, em produção real
  (streaming, sem acesso ao futuro), não pode ser calculada da mesma forma. O desempenho medido em
  `treino`/`teste` está inflado em relação ao que a planta consegue de fato calcular hora a hora.

### 3. `maquina_risco` usa o próprio rótulo do lote (vazamento de alvo)

Quando o lote contém `falha_72h`, `_risco_por_maquina` calcula o risco como a média do próprio
`falha_72h`, por máquina, dentro do mesmo lote sendo avaliado.

- **Teste que falha:** `tests/test_features.py::test_maquina_risco_nao_deveria_usar_o_proprio_rotulo_do_lote_bug`
- **Evidência medida:** correlação de 0,50 entre `maquina_risco` e `falha_72h` no conjunto de teste;
  é também a feature com maior impacto médio na probabilidade (Bloco C, ~0,11, quase o dobro da
  segunda colocada).
- **Causa raiz:** a condição `if "falha_72h" in df.columns"` em `_risco_por_maquina` — o comportamento
  é intencional no código, mas metodologicamente ilegítimo para avaliação: só existe porque a falha
  já é conhecida no lote.
- **Impacto na planta:** todo número reportado sobre desempenho em `treino`/`teste` (incluindo a
  comparação v1×v2 do Bloco B) está inflado por essa fuga de rótulo. Em `producao`, sem rótulo, a
  função cai para a tabela congelada — ou seja, o modelo é avaliado com uma vantagem que ele não tem
  na vida real. Corrigir isso de verdade exige retreinar os modelos sem essa fuga, fora do escopo do
  time de testes.

### 4. `Modelo.prever_registro` ignora as chaves do dicionário

O método monta o vetor de entrada com `[float(v) for v in registro.values()]`, na ordem de inserção
do dicionário Python, sem validar as chaves contra `ordem_features`.

- **Teste que falha:** `tests/test_pipeline.py::test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug`
- **Evidência medida:** reembaralhar as chaves de 200 registros (mantendo os mesmos valores) muda a
  decisão em cerca de 32,5% dos casos (número exato no log de execução), com variação de
  probabilidade de até ~0,90.
- **Causa raiz:** ausência de validação de chave; o método confia que quem o chama sempre monta o
  dicionário na ordem exata de `ordem_features`, sem documentar essa exigência nem falhar quando ela
  é violada.
- **Impacto na planta:** é o caminho usado pelo painel da sala de controle, que recebe um registro por
  vez. Qualquer mudança de ordem na construção desse dicionário — um reordenamento de campos no
  formulário, uma serialização diferente — muda decisões de manutenção silenciosamente, sem erro,
  sem log, sem aviso.

## Outros achados relevantes (não são bugs de código, mas afetam a decisão de negócio)

- **v2 troca recall por acurácia.** v2 tem +3,5 pp de acurácia sobre v1 (confirmado com IC 95%:
  +2,7 a +4,3 pp), mas o recall cai de 94,2% para 81,4% (IC 95%: -15,3 a -10,3 pp) — o número de
  falsos negativos triplica (38 → 122 no conjunto de teste). McNemar confirma que as decisões
  divergem de forma sistemática (p ≈ 0). Promover a v2 porque ela "ganhou 3,5 pontos" ignora que ela
  perde quase 1 em cada 6 motores que realmente vão falhar.
- **O limiar 0,5 não maximiza F1.** No conjunto de teste, o limiar que maximiza F1 para v1 fica perto
  de 0,7 (F1 ≈ 0,79 contra 0,72 em 0,5) — mas dado o custo maior de um falso negativo, a escolha certa
  de limiar depende de quanto a planta está disposta a pagar em falsos positivos, não só de F1.
- **As probabilidades da v1 não são calibradas.** Nas faixas de probabilidade baixa e média
  (0,1-0,4), a probabilidade média prevista chega a ser ~30x maior que a frequência real observada de
  falha — provavelmente efeito da reponderação de classe usada no treino. Só nas faixas altas
  (0,8-1,0) a calibração é razoável.
- **Sem drift prático relevante entre teste e produção.** O teste de Kolmogorov-Smirnov por coluna
  encontra diferença estatisticamente significativa (esperado, dado o tamanho da amostra), mas o
  tamanho do efeito (estatística KS) fica abaixo de 0,15 em todas as colunas testadas — a diferença de
  distribuição é real, mas pequena demais para explicar sozinha uma queda de desempenho em produção.

## Correções aplicadas (bônus)

Ver `correcoes/README.md` — patches para os defeitos 1, 2 e 4, com o teste correspondente passando a
verde contra o código corrigido. O defeito 3 fica só documentado (exige retreinar os modelos).
```

- [ ] **Step 2: Commit**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
git add relatorio.md
git commit -m "Escreve relatorio com os achados, evidencia, causa raiz e impacto"
```

---

### Task 9: Checklist final de pronto

**Files:** nenhum arquivo novo — apenas verificação.

- [ ] **Step 1: Rodar a suíte inteira uma última vez, do zero, num shell novo**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
./.venv/Scripts/python.exe -m pytest -v
```

Expected: mesma composição de PASSED/FAILED da Task 6 — nada quebrou entre as tasks 6 e 8.

- [ ] **Step 2: Confirmar que o pacote `sentinela` não foi editado**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela"
git status
git log --oneline -3
```

Expected: `working tree clean`, branch `main`, sem a branch `correcao/defeitos-confirmados` deixada
como corrente (pode existir como branch, só não pode estar checked out nem ter ido para `main`).

- [ ] **Step 3: Confirmar a lista de entrega contra o guia do aluno**

- [ ] Repositório com `tests/`, `README.md`, `requirements.txt`, `relatorio.md`
- [ ] Suíte rodando de verdade, log colado no README
- [ ] Pelo menos uma falha real documentada (são 8: seção "Defeitos confirmados" cobre 4 delas em
  detalhe; as demais — limiar, calibração, ruído sub-sensor, importância de `maquina_risco` — estão
  nos achados adicionais)
- [ ] Bloco A, B e C todos representados
- [ ] Correção opcional com teste antes/depois

- [ ] **Step 4: Commit final (se sobrou algo solto)**

```bash
cd "C:/Users/User/Documents/repositorio/sentinela-testes"
git status
git add -A
git diff --cached --quiet || git commit -m "Ajustes finais de entrega"
```

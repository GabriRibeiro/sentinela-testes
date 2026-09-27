# Suíte de testes do Sentinela — design

Data: 2026-09-27
Trilha: 1 (ML clássico) — Projeto Final, Testes Automatizados para Modelos de IA, IEC PUC Minas
Sistema sob teste: https://github.com/felipehp/sentinela-nortemec (pacote `sentinela`, não editado)
Autoria: individual

## Objetivo

Escrever a suíte de testes que o pacote `sentinela` nunca teve, cobrindo os três blocos obrigatórios
da trilha (unitário, estatístico, adversarial), e documentar pelo menos uma falha real do sistema
com evidência medida. Entrega: repositório com `pytest`, README com instruções de execução e log
colado, e `relatorio.md` com os achados.

## Ambiente

- Ambiente virtual Python (`venv`) próprio deste repositório, isolado do resto da máquina.
- `sentinela` clonado e instalado em modo editável (`pip install -e .`) a partir do repositório
  oficial, como dependência — nunca editado.
- Sem necessidade de virtualização de hardware (Docker/WSL/VM): tudo roda nativamente via
  Python + pytest.

## Estrutura do repositório

```
sentinela-testes/
  README.md
  requirements.txt
  pytest.ini
  tests/
    conftest.py
    test_preprocessamento.py
    test_features.py
    test_pipeline.py
    test_estatistico.py
    test_adversarial.py
  correcoes/
    README.md
    *.patch
  relatorio.md
```

## Defeitos confirmados na exploração (evidência já medida)

1. **Pressão em unidade mista.** ~32% das linhas chegam em psi (valores ~48–65) e
   `preprocessamento.limpar` nunca converte para bar (~3–4,5). A feature `pressao` mistura duas
   escalas sem aviso, violando a ficha técnica do README.
2. **Vazamento temporal em `temp_media_6h`.** `features.construir` usa
   `rolling(JANELA_CURTA, center=True)`, que depende de leituras futuras. Medido: truncar a série
   de uma máquina no instante *t* muda o valor da feature em *t* (72,69 → 72,57 no caso testado).
   Viola a garantia de que o valor no instante *t* só depende de leituras até *t*.
3. **Vazamento do rótulo em `maquina_risco`.** Quando o lote contém `falha_72h`,
   `_risco_por_maquina` calcula o risco como a média do próprio rótulo por máquina, no mesmo lote
   que está sendo avaliado. Correlação medida entre `maquina_risco` e `falha_72h` no conjunto de
   teste: 0,50. Isso infla artificialmente o desempenho medido em `teste` (que tem rótulo) frente a
   `producao` real (que usa a tabela congelada).
4. **`Modelo.prever_registro` ignora as chaves do dicionário.** O método monta o vetor de entrada a
   partir de `dict.values()`, na ordem de inserção do dicionário, sem validar as chaves contra
   `ordem_features`. Medido: reembaralhar as chaves de um registro (mantendo os mesmos valores)
   muda a decisão em 32,5% de 200 casos testados, com variação de probabilidade de até 0,90. É o
   caminho usado pelo painel da sala de controle.

## Blocos de teste

### Bloco A — unitários (`test_preprocessamento.py`, `test_features.py`, `test_pipeline.py`)

- Tipos e normalização produzidos por `limpar()` (timestamp, ids, unidade_pressao, turno, idade).
- Tratamento de faltantes: `vibracao_rms` preenchida com 0,0 — fora da faixa de operação
  (1,2–8,0 mm/s) da ficha técnica; documentar como valor fisicamente implausível.
- Faixas e unidades do contrato de dados por coluna, validando os dados crus em `dados/`, não só o
  código — é aqui que o bug de pressão psi/bar aparece.
- Ordem das linhas preservada: embaralhar a entrada e comparar com a saída.
- Uso só de leituras até o instante corrente: comparar cada feature de janela calculada com a série
  truncada em *t* vs. a série completa, para todas as features de janela — expõe o vazamento de
  `temp_media_6h` e serve de controle para as demais (que devem passar).
- Recorte vs. lote inteiro: mesmas linhas, recorte pequeno vs. lote inteiro — expõe também o
  vazamento de `maquina_risco` quando o lote tem rótulo.
- Contrato de `Modelo.prever_registro`: teste de reembaralhamento de chaves medindo a taxa de
  decisões diferentes para o mesmo registro.

### Bloco B — estatísticos (`test_estatistico.py`)

- Prevalência da classe positiva (~15%) e por que acurácia não responde à pergunta da manutenção;
  métrica adequada (precisão/recall/F1/PR-AUC).
- Comparação v1 × v2 com incerteza: bootstrap (IC) e McNemar sobre as decisões pareadas no conjunto
  de teste. Repetir a comparação recalculando `maquina_risco` pela tabela congelada (simulando
  produção real, sem vazamento) para checar se a vantagem da v2 se sustenta.
- Varredura do limiar de decisão (precisão × recall em função do limiar, não só em 0,5).
- Calibração: probabilidade média prevista vs. frequência observada de falha, por faixa de
  probabilidade — checar se a reponderação de classe da v1 descalibra as probabilidades.
- Comparação de distribuições `teste` × `producao` por coluna via teste de Kolmogorov-Smirnov, não
  só comparação de médias.

### Bloco C — adversariais (`test_adversarial.py`)

- Perturbação de temperatura menor que o ruído do sensor (±1,0 °C): taxa de decisões que mudam.
- Perturbação feature a feature: quais mais pesam na decisão, e se alguma dependência é
  causalmente ilegítima (`maquina_risco`, `operador_senior`, `turno`).
- Testes contrafactuais: alterar `id_operador`/`turno` mantendo o estado físico do motor — a decisão
  deveria se manter.
- Casos-limite: leituras fora da faixa da ficha técnica, lote com um motor só, lote com uma linha só.

## Correções (bônus, opcional)

Para os defeitos 1, 2 e 4 — os de correção mecânica — um patch `.patch` por item em `correcoes/`,
gerado a partir de uma branch separada no clone do `sentinela` (nunca aplicado ao pacote usado pelo
resto da suíte), com o teste que falhava antes rodando de novo, agora passando, contra a cópia
corrigida. O defeito 3 (vazamento em `maquina_risco`) entra só como achado discutido no relatório:
corrigi-lo de verdade exigiria retreinar os modelos, fora do escopo do time de testes.

## Critério de pronto

- `pytest -v` roda a suíte inteira sem erros de coleta; testes que documentam defeito falham de
  propósito e estão marcados/comentados como tal; os demais passam.
- README com passo a passo de instalação e execução, e o log real da suíte colado.
- `relatorio.md` com, para cada defeito confirmado: teste que falha, evidência medida, causa raiz,
  impacto na planta.

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

## Como foi verificado

1. Branch `correcao/defeitos-confirmados` criada em `../sentinela`, a partir de `main`.
2. Os três arquivos corrigidos, commitados nessa branch.
3. `git diff main -- <arquivo>` gerou cada `.patch` a partir dessa branch.
4. O pacote corrigido foi reinstalado no venv deste repositório
   (`pip install -e ../sentinela --force-reinstall --no-deps`) e os três testes acima passaram a
   PASSED — log em `../docs/log_correcoes.txt`.
5. `../sentinela` voltou para `main` (`git checkout main`) e o pacote original foi reinstalado no
   venv, devolvendo os três testes ao estado FAILED esperado pelo resto da suíte.

`../sentinela` fica, ao final, sem nenhuma edição em `main` — a branch de correção existe só como
histórico local, nunca mesclada.

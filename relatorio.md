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
- **Evidência medida:** no conjunto de teste, 100% das leituras em psi ficam fora da faixa de
  operação (3,0-4,5 bar) depois de `limpar()` — a mensagem de falha traz min/max/média de `pressao`
  por `unidade_pressao` original, com o grupo psi na faixa 48,4-64,7.
- **Causa raiz:** falta a linha de conversão (1 bar = 14,5038 psi) em `limpar()`; a normalização de
  string (`.str.lower()`) engana quem lê o código, pois parece que a unidade "foi tratada".
- **Impacto na planta:** a feature `pressao` alimenta o modelo com dois significados físicos
  diferentes para o mesmo número — para ~32% dos motores, a "pressão" que o modelo vê está cerca de
  15x maior que a pressão real do motor. Qualquer relação aprendida entre pressão e falha está
  contaminada por essa mistura de escalas.
- **Correção:** aplicada em `correcoes/01-pressao-psi-bar.patch` (ver `correcoes/README.md`).

### 2. `temp_media_6h` depende de leituras futuras (look-ahead)

`features.construir` calcula `temp_media_6h` com `rolling(JANELA_CURTA, center=True, min_periods=1)`.
`center=True` usa leituras depois do instante *t* para compor o valor em *t*.

- **Teste que falha:** `tests/test_features.py::test_temp_media_6h_depende_do_futuro_bug`
- **Evidência medida:** truncar a série da máquina M01 no instante t=20 muda o valor de
  `temp_media_6h` de 72,69 (com o futuro disponível) para 72,57 (só com o passado) — dois valores
  diferentes para o mesmo instante. As outras quatro features de janela testadas com o mesmo método
  (`temp_max_24h`, `delta_temp_24h`, `vib_media_6h`, `vib_max_24h`, `corrente_media_6h`) não mudam.
- **Causa raiz:** `center=True` no `rolling`; é a única das seis features de janela que usa esse
  parâmetro.
- **Impacto na planta:** o modelo foi treinado e avaliado com uma feature que, em produção real
  (streaming, sem acesso ao futuro), não pode ser calculada da mesma forma. O desempenho medido em
  `treino`/`teste` está inflado em relação ao que a planta consegue de fato calcular hora a hora.
- **Correção:** aplicada em `correcoes/02-temp-media-6h-look-ahead.patch`.

### 3. `maquina_risco` usa o próprio rótulo do lote (vazamento de alvo)

Quando o lote contém `falha_72h`, `_risco_por_maquina` calcula o risco como a média do próprio
`falha_72h`, por máquina, dentro do mesmo lote sendo avaliado.

- **Teste que falha:** `tests/test_features.py::test_maquina_risco_nao_deveria_usar_o_proprio_rotulo_do_lote_bug`
- **Evidência medida:** correlação de 0,504 entre `maquina_risco` e `falha_72h` no conjunto de
  teste. No Bloco C, é também a feature com maior impacto médio na probabilidade prevista pela v1
  (0,1097 — mais do que o dobro da segunda colocada, `horas_operacao`, com 0,0647).
- **Causa raiz:** a condição `if "falha_72h" in df.columns` em `_risco_por_maquina` — o comportamento
  é intencional no código, mas metodologicamente ilegítimo para avaliação: só existe porque a falha
  já é conhecida no lote.
- **Impacto na planta:** todo número reportado sobre desempenho em `treino`/`teste` (incluindo a
  comparação v1×v2 do Bloco B) está inflado por essa fuga de rótulo. Em `producao`, sem rótulo, a
  função cai para a tabela congelada — ou seja, o modelo é avaliado com uma vantagem que não tem na
  vida real.
- **Correção:** não aplicada. Corrigir de verdade exige mudar a definição da feature e retreinar os
  dois modelos, fora do escopo de um time de testes.

### 4. `Modelo.prever_registro` ignora as chaves do dicionário

O método monta o vetor de entrada com `[float(v) for v in registro.values()]`, na ordem de inserção
do dicionário Python, sem validar as chaves contra `ordem_features`.

- **Teste que falha:** `tests/test_pipeline.py::test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug`
- **Evidência medida:** reembaralhar as chaves de 200 registros (mantendo os mesmos valores) muda a
  decisão em 65 casos (32,5%).
- **Causa raiz:** ausência de validação de chave; o método confia que quem o chama sempre monta o
  dicionário na ordem exata de `ordem_features`, sem documentar essa exigência nem falhar quando ela
  é violada.
- **Impacto na planta:** é o caminho usado pelo painel da sala de controle, que recebe um registro por
  vez. Qualquer mudança de ordem na construção desse dicionário — um reordenamento de campos no
  formulário, uma serialização diferente — muda decisões de manutenção silenciosamente, sem erro,
  sem log, sem aviso.
- **Correção:** aplicada em `correcoes/03-prever-registro-ordem-chaves.patch`.

## Outros achados relevantes (não são bugs de código, mas afetam a decisão de negócio)

- **v2 troca recall por acurácia.** v2 tem +3,5 pp de acurácia sobre v1 no conjunto de teste
  (88,6% → 92,1%, IC 95% bootstrap para a diferença: +2,7 a +4,3 pp), mas o recall cai de 94,2% para
  81,4% (IC 95%: -15,3 a -10,3 pp) — o número de falsos negativos triplica (38 → 122 em 4.200
  leituras). O teste de McNemar confirma que as decisões divergem de forma sistemática (p ≈ 0), não
  por acaso. Promover a v2 porque ela "ganhou 3,5 pontos" ignora que ela perde quase 1 em cada 6
  motores que realmente vão falhar — e, dado o vazamento do achado 3, mesmo esse número pode estar
  otimista demais.
- **O limiar 0,5 não maximiza F1.** No conjunto de teste, o limiar que maximiza F1 para v1 fica em
  0,7 (F1 = 0,786, contra 0,720 em 0,5). Isso não significa que 0,7 seja o limiar certo — dado que um
  falso negativo custa mais que um falso positivo, a escolha ideal provavelmente pesa para o lado do
  recall, não do F1 — mas mostra que o limiar em produção nunca foi de fato otimizado, só herdado.
- **As probabilidades da v1 não são calibradas.** Nas faixas de probabilidade baixa e média
  (0,1-0,4), o maior desvio encontrado é na faixa 0,3-0,4: probabilidade média prevista de 0,341
  contra frequência real observada de apenas 0,003 — quase 100x mais pessimista que a realidade
  nessa faixa. Só nas faixas altas (0,8-1,0) a calibração é razoável. Efeito provável da
  reponderação de classe usada no treino da v1.
- **Sem drift prático relevante entre teste e produção.** Comparando as distribuições por
  Kolmogorov-Smirnov (não só a média, que pode escondar diferença de forma), a estatística KS fica
  abaixo de 0,15 em todas as cinco colunas testadas (temperatura, vibração, pressão, corrente, rpm)
  — a maior é temperatura, com KS=0,13. A diferença é estatisticamente detectável (esperado, dado o
  tamanho da amostra), mas pequena demais para explicar sozinha uma eventual queda de desempenho em
  produção.
- **Ruído sub-sensor muda 0,38% das decisões.** Perturbar a temperatura em até ±0,5°C — metade do
  ruído declarado do próprio sensor (±1,0°C) — já é suficiente para mudar a decisão em 16 de 4.200
  leituras. Pequeno, mas real: parte da robustez do sistema depende de sorte na casa decimal.
- **Turno, isoladamente, não muda a decisão.** Trocar só o turno (1/2/3), mantendo o resto do estado
  físico do motor igual, não mudou a decisão em nenhum dos 300 casos testados — ao contrário de
  `maquina_risco`, essa dependência não se mostrou um problema.

## Correções aplicadas (bônus)

Ver `correcoes/README.md` — patches para os defeitos 1, 2 e 4, com o teste correspondente passando a
verde contra o código corrigido e voltando a vermelho contra o código original. O defeito 3 fica só
documentado (exige retreinar os modelos).

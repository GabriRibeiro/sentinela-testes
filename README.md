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

A suíte tem 49 testes: 38 que devem passar e 11 que devem falhar de propósito — cada falha
documenta um defeito real do sistema, com o número medido na própria mensagem de asserção. Ver
`relatorio.md` para a análise de cada um.

## Log de execução real

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\User\Documents\repositorio\sentinela-testes\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\User\Documents\repositorio\sentinela-testes
configfile: pytest.ini
testpaths: tests
collecting ... collected 49 items

tests/test_adversarial.py::test_perturbacao_menor_que_o_ruido_do_sensor_nao_deveria_mudar_a_decisao_bug FAILED [  2%]
tests/test_adversarial.py::test_maquina_risco_e_a_dependencia_mais_forte_e_e_causalmente_ilegitima_bug FAILED [  4%]
tests/test_adversarial.py::test_contrafactual_turno_nao_deveria_mudar_a_decisao PASSED [  6%]
tests/test_adversarial.py::test_lote_com_um_motor_so_nao_quebra PASSED   [  8%]
tests/test_adversarial.py::test_leitura_extrema_fora_da_ficha_tecnica_nao_quebra_a_predicao PASSED [ 10%]
tests/test_estatistico.py::test_prevalencia_e_baseline_trivial_mostram_que_acuracia_engana PASSED [ 12%]
tests/test_estatistico.py::test_v2_tem_mais_acuracia_mas_recall_pior_que_v1 PASSED [ 14%]
tests/test_estatistico.py::test_bootstrap_ic_confirma_que_a_perda_de_recall_da_v2_nao_e_ruido PASSED [ 16%]
tests/test_estatistico.py::test_mcnemar_confirma_que_v1_e_v2_discordam_de_forma_nao_aleatoria PASSED [ 18%]
tests/test_estatistico.py::test_limiar_0_5_nao_maximiza_f1_bug FAILED    [ 20%]
tests/test_estatistico.py::test_probabilidades_da_v1_nao_sao_calibradas_na_faixa_baixa_e_media_bug FAILED [ 22%]
tests/test_estatistico.py::test_distribuicao_teste_e_producao_sem_drift_pratico_relevante[temperatura_c] PASSED [ 24%]
tests/test_estatistico.py::test_distribuicao_teste_e_producao_sem_drift_pratico_relevante[vibracao_rms] PASSED [ 26%]
tests/test_estatistico.py::test_distribuicao_teste_e_producao_sem_drift_pratico_relevante[pressao] PASSED [ 28%]
tests/test_estatistico.py::test_distribuicao_teste_e_producao_sem_drift_pratico_relevante[corrente_a] PASSED [ 30%]
tests/test_estatistico.py::test_distribuicao_teste_e_producao_sem_drift_pratico_relevante[rpm] PASSED [ 32%]
tests/test_features.py::test_construir_devolve_exatamente_as_colunas_de_ordem_features PASSED [ 34%]
tests/test_features.py::test_construir_preserva_a_ordem_das_linhas_de_entrada PASSED [ 36%]
tests/test_features.py::test_construir_da_o_mesmo_valor_independente_da_ordem_de_entrada PASSED [ 38%]
tests/test_features.py::test_features_de_janela_controle_nao_dependem_do_futuro[temp_max_24h] PASSED [ 40%]
tests/test_features.py::test_features_de_janela_controle_nao_dependem_do_futuro[delta_temp_24h] PASSED [ 42%]
tests/test_features.py::test_features_de_janela_controle_nao_dependem_do_futuro[vib_media_6h] PASSED [ 44%]
tests/test_features.py::test_features_de_janela_controle_nao_dependem_do_futuro[vib_max_24h] PASSED [ 46%]
tests/test_features.py::test_features_de_janela_controle_nao_dependem_do_futuro[corrente_media_6h] PASSED [ 48%]
tests/test_features.py::test_temp_media_6h_depende_do_futuro_bug FAILED  [ 51%]
tests/test_features.py::test_maquina_risco_nao_deveria_usar_o_proprio_rotulo_do_lote_bug FAILED [ 53%]
tests/test_features.py::test_features_sem_janela_sao_iguais_em_recorte_e_no_lote_inteiro[pressao] PASSED [ 55%]
tests/test_features.py::test_features_sem_janela_sao_iguais_em_recorte_e_no_lote_inteiro[rpm] PASSED [ 57%]
tests/test_features.py::test_features_sem_janela_sao_iguais_em_recorte_e_no_lote_inteiro[idade_equipamento_meses] PASSED [ 59%]
tests/test_features.py::test_features_sem_janela_sao_iguais_em_recorte_e_no_lote_inteiro[horas_operacao] PASSED [ 61%]
tests/test_features.py::test_features_sem_janela_sao_iguais_em_recorte_e_no_lote_inteiro[operador_senior] PASSED [ 63%]
tests/test_features.py::test_features_sem_janela_sao_iguais_em_recorte_e_no_lote_inteiro[turno] PASSED [ 65%]
tests/test_pipeline.py::test_prever_registro_concorda_com_prever_proba_quando_a_ordem_esta_certa PASSED [ 67%]
tests/test_pipeline.py::test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug FAILED [ 69%]
tests/test_pipeline.py::test_pipeline_executar_fim_a_fim PASSED          [ 71%]
tests/test_pipeline.py::test_artefatos_e_dados_batem_com_o_manifest PASSED [ 73%]
tests/test_pipeline.py::test_lote_com_um_motor_so PASSED                 [ 75%]
tests/test_pipeline.py::test_lote_com_uma_linha_so PASSED                [ 77%]
tests/test_pipeline.py::test_leitura_fora_da_faixa_da_ficha_tecnica_nao_quebra_a_predicao PASSED [ 79%]
tests/test_preprocessamento.py::test_limpar_normaliza_tipos_e_rotulos PASSED [ 81%]
tests/test_preprocessamento.py::test_limpar_nao_modifica_o_quadro_de_entrada PASSED [ 83%]
tests/test_preprocessamento.py::test_motor_parado_e_removido PASSED      [ 85%]
tests/test_preprocessamento.py::test_vibracao_faltante_e_preenchida_mas_fora_da_faixa_de_operacao_bug FAILED [ 87%]
tests/test_preprocessamento.py::test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug[treino] FAILED [ 89%]
tests/test_preprocessamento.py::test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug[teste] FAILED [ 91%]
tests/test_preprocessamento.py::test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug[producao] FAILED [ 93%]
tests/test_preprocessamento.py::test_demais_colunas_respeitam_a_ficha_tecnica_dos_sensores[treino] PASSED [ 95%]
tests/test_preprocessamento.py::test_demais_colunas_respeitam_a_ficha_tecnica_dos_sensores[teste] PASSED [ 97%]
tests/test_preprocessamento.py::test_demais_colunas_respeitam_a_ficha_tecnica_dos_sensores[producao] PASSED [100%]

================================== FAILURES ===================================
_ test_perturbacao_menor_que_o_ruido_do_sensor_nao_deveria_mudar_a_decisao_bug _
E       AssertionError: BUG: perturbar a temperatura em até ±0.5°C — metade do ruído declarado do sensor (±1.0°C, README) — muda a decisão em 0.38% das leituras. Uma ordem de manutenção não devia depender de uma variação menor que a precisão do próprio instrumento.
E       assert np.float64(0.0038095238095238095) == 0

_ test_maquina_risco_e_a_dependencia_mais_forte_e_e_causalmente_ilegitima_bug _
E       AssertionError: BUG: a feature que mais move a probabilidade quando deslocada em 1 desvio-padrão é 'maquina_risco' (impacto médio 0.1097, o dobro da segunda colocada). Ela não descreve o estado físico atual do motor — é um resumo do histórico de falhas da própria máquina, incluindo, no lote rotulado, o rótulo que está sendo previsto.
E       assert 'maquina_risco' not in {'maquina_risco'}

_____________________ test_limiar_0_5_nao_maximiza_f1_bug _____________________
E       AssertionError: BUG (decisão de projeto, não de código): limiar 0.5 dá F1=0.720; limiar 0.7 dá F1=0.786 — o limiar em produção não é o que maximiza F1 no conjunto de teste.
E       assert 0.7862903225806451 <= (0.7203736135434909 + 1e-09)

___ test_probabilidades_da_v1_nao_sao_calibradas_na_faixa_baixa_e_media_bug ___
E       AssertionError: BUG: na faixa [0.3-0.4], a v1 prevê probabilidade média 0.341 mas a frequência real observada é 0.003 (erro de 0.338) — a reponderação de classe no treino provavelmente distorce a probabilidade, que deixa de ser uma probabilidade de verdade nessa faixa.
E       assert np.float64(0.33751327970011535) < 0.1

__________________ test_temp_media_6h_depende_do_futuro_bug ___________________
E       AssertionError: BUG: temp_media_6h usa rolling(JANELA_CURTA, center=True), que olha leituras futuras. No instante t=20, o valor muda de 72.57249999999999 (só com o passado) para 72.68666666666667 (com o futuro disponível) — a mesma leitura, dois resultados diferentes.
E       assert np.float64(72.57249999999999) == 72.68666666666667 ± 7.3e-05
E
E         comparison failed
E         Obtained: 72.57249999999999
E         Expected: 72.68666666666667 ± 7.3e-05

______ test_maquina_risco_nao_deveria_usar_o_proprio_rotulo_do_lote_bug _______
E       AssertionError: BUG: quando o lote tem falha_72h, maquina_risco é a média do próprio rótulo por máquina, dentro do mesmo lote sendo avaliado (correlação medida com o rótulo: 0.504). Isso é a variável usando informação que só existe porque a falha já é conhecida.
E       assert np.float64(0.5042055500332061) < 0.15
E        +  where np.float64(0.5042055500332061) = abs(np.float64(0.5042055500332061))

______ test_prever_registro_ignora_a_ordem_das_chaves_do_dicionario_bug _______
E       AssertionError: BUG: prever_registro monta o vetor de entrada a partir de dict.values(), na ordem de inserção do dicionário, sem validar as chaves contra ordem_features. Reembaralhar as chaves do mesmo registro (mesmos valores) muda a decisão em 65/200 casos (32.5%). É o caminho usado pelo painel da sala de controle.
E       assert 65 == 0

____ test_vibracao_faltante_e_preenchida_mas_fora_da_faixa_de_operacao_bug ____
E       AssertionError: BUG: vibracao_rms ausente é preenchida com 0.0 (VIBRACAO_PADRAO), um valor fora da faixa de operação do sensor (1.2-8.0 mm/s). Valor mínimo obtido após limpar(): 0.0.
E       assert np.float64(0.0) >= 1.2

__ test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug[treino] ___
E       AssertionError: BUG: em 'treino', limpar() nunca converte pressão de psi para bar. Faixas por unidade após a limpeza:
E                             min     max       mean
E         unidade_pressao
E         bar               3.179   4.565   3.941830
E         psi              48.626  65.320  57.415325
E         A ficha técnica exige 3.0-4.5 bar para todas as linhas, independente da unidade original.
E       assert False

___ test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug[teste] ___
E       AssertionError: BUG: em 'teste', limpar() nunca converte pressão de psi para bar. Faixas por unidade após a limpeza:
E                             min     max       mean
E         unidade_pressao
E         bar               3.205   4.433   3.901291
E         psi              48.362  64.679  56.509528
E         A ficha técnica exige 3.0-4.5 bar para todas as linhas, independente da unidade original.
E       assert False

_ test_pressao_apos_limpar_respeita_a_faixa_de_operacao_em_bar_bug[producao] __
E       AssertionError: BUG: em 'producao', limpar() nunca converte pressão de psi para bar. Faixas por unidade após a limpeza:
E                             min     max       mean
E         unidade_pressao
E         bar               3.264   4.511   3.933440
E         psi              47.662  64.167  56.021307
E         A ficha técnica exige 3.0-4.5 bar para todas as linhas, independente da unidade original.
E       assert False

======================= 11 failed, 38 passed in 17.23s ========================
```

Log completo (com traceback inteiro de cada falha): `docs/log_execucao.txt`.

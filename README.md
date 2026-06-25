# Otimização de Treinamento de Redes Neurais via Controle PID

Comparação de quatro otimizadores (SGD, Adam, SGD-PID e Adam-PID) num MLP
sobre a base MNIST, com medição de tempo, FLOPs e estimativa de energia/CO2.
Código em NumPy puro: cada regra de atualização é escrita diretamente a partir
das equações de controle dos artigos de referência.

## Arquivos
- `fig_curvas.png`, `fig_convergencia.png`, `fig_eficiencia.png` — figuras
- `core.py`            — MLP + os 4 otimizadores + laço de treino (mede tempo e FLOPs)
- `run_experiment.py`  — treina os otimizadores e mede energia (CodeCarbon); salva results.json
- `make_figs.py`       — gera as figuras de curvas/convergência e a tabela de acurácia
- `make_efficiency.py` — gera a tabela e a figura de eficiência (tempo/energia/FLOPs)
- `baixar_mnist.py`    — baixa a base MNIST (recria mnist.npz)
- `results.json`       — métricas registradas época a época

## Como reproduzir
```bash
pip install -r requirements.txt
python baixar_mnist.py
python run_experiment.py SGD Adam SGD-PID Adam-PID   # ~7 min em 1 CPU
python make_figs.py
python make_efficiency.py
```
## Métricas de eficiência
- **Tempo de computo** e **FLOPs**: medidos diretamente em core.py.
- **Energia (kWh) e CO2 (g)**: estimados com CodeCarbon, usando o fator de
  emissão da rede elétrica brasileira (~0,12 kg CO2/kWh).
- A métrica-chave é o custo *até atingir* uma acurácia-alvo, não o custo total.

Obs.: sem contadores RAPL/GPU, a energia é uma *estimativa* baseada na potência
da CPU. Tempo e FLOPs são exatos.

## Otimizadores
- SGD      -> controlador P
- Adam     -> controlador I adaptativo
- SGD-PID  -> P + I + D (An et al., CVPR 2018)
- Adam-PID -> Adam + P + D (Dai et al., IET CTA 2023); com Kp=0,Ki=1,Kd=0 vira Adam

# Otimização do Treinamento de Redes Neurais: uma Abordagem por Controladores PID

Avaliação crítica e reprodutível de otimizadores baseados em gradiente
reinterpretados como controladores **PID** (proporcional–integral–derivativo),
com medição de **energia e emissões de CO₂** na matriz elétrica brasileira.

Autora: **Laura Inês Freitas Campos** — Laboratório de IA, CEFET-MG
(orientação: Prof. Everthon de Souza Oliveira).

---

## Sobre o projeto

Trabalhos recentes ([An *et al.*, 2018](#referências); [Dai *et al.*, 2023](#referências))
observam que a atualização de parâmetros por gradiente é análoga à ação de um
controlador realimentado, com o gradiente no papel do *erro*. Sob essa ótica, o
**SGD** é um controlador proporcional (P), o **Adam** é um controlador integral
adaptativo (I), e os métodos **SGD-PID** e **Adam-PID** completam o controlador
acrescentando os termos que faltam.

Este repositório implementa e compara quatro otimizadores — **SGD, Adam,
SGD-PID e Adam-PID** — sob um protocolo de **comparação justa** (sintonização de
hiperparâmetros para todos os métodos, inclusive os *baselines*; avaliação com
múltiplas sementes e testes estatísticos), em três regimes: **MLP/MNIST**,
**MLP/Fashion-MNIST** e **CNN/CIFAR-10**.

### Principais achados

- Sob comparação justa, a incorporação dos termos P e D ao Adam **não** produz
  ganho de acurácia estatisticamente significativo em nenhum dos regimes
  (teste *t* de Welch, *p* > 0,05).
- O **termo derivativo é inerte** no regime de mini-lotes: a variação do
  gradiente entre passos é dominada por ruído de amostragem.
- A vantagem de eficiência da abordagem PID é **dependente de regime**: só
  compensa quando a convergência mais rápida economiza muitas épocas.
- Disponibiliza-se uma **implementação aberta de referência** dos quatro
  otimizadores — em particular do Adam-PID, ausente na literatura.

---

## Estrutura do repositório

```
.
├── core.py                # MLP + 4 otimizadores em NumPy puro (implementação pedagógica)
├── conv.py                # CNN em NumPy com im2col (regime CIFAR-10)
├── pid_optimizers.py      # Otimizadores PID em PyTorch: SGD_PID, SGD_PID_Classic, Adam_PID
├── compare_mnist_mlp.py   # Harness MLP em GPU + registro de datasets (mnist/fashion/emnist)
├── tune_mnist.py          # Sintonização e avaliação (fases: grid, confirm, dterm, sgdpid, final, all)
├── run_experiment.py      # Experimento CNN/CIFAR-10 com medição de energia (CodeCarbon)
├── make_figs_mlp.py       # Figuras dos regimes MLP (curvas, barras, termo D, energia)
├── make_figs.py           # Figuras do regime CNN/CIFAR-10
├── make_efficiency.py     # Métricas de eficiência do regime CNN/CIFAR-10
├── tune_pid.py            # Sintonização da Fase 1 (CIFAR, NumPy)
├── confirm_seeds.py       # Confirmação multi-semente da Fase 1 (CIFAR)
├── requirements.txt
├── results.json           # Resultados CNN/CIFAR-10 (com energia e CO₂)
├── confirm_seeds.json     # Adam vs. Adam-PID, multi-semente (CIFAR-10)
├── experiments/
│   ├── tune_mnist/        # JSONs de resultado do MNIST (grid, confirm, dterm, sgdpid, final)
│   └── tune_fashion/      # JSONs de resultado do Fashion-MNIST
├── imagens/
│   ├── fase1/             # Figuras da Fase 1
│   └── fase2/             # Figuras da Fase 2
└── old/                   # Arquivos descontinuados (mantidos por histórico)
```

**Duas implementações independentes.** Os otimizadores existem em NumPy puro
(`core.py`, `conv.py`), em que cada passo corresponde literalmente às equações do
artigo, e em PyTorch/GPU (`pid_optimizers.py`), usado nos experimentos de maior
escala. A equivalência foi verificada: o Adam-PID reduz-se ao Adam (diferença da
ordem de 10⁻¹⁷) quando `Kp=0, Ki=1, Kd=0`.

---

## Requisitos e instalação

Requer **Python 3.10+** e, para os experimentos acelerados, uma **GPU NVIDIA**
com CUDA (desenvolvido em uma GTX 1650). Os experimentos rodam em CPU, porém
mais lentamente.

```bash
# 1. clonar o repositório
git clone https://github.com/laurafreitascampos/analise-de-otimizadores-ML.git
cd analise-de-otimizadores-ML

# 2. criar e ativar o ambiente virtual
python -m venv venv
source venv/bin/activate          # bash/zsh
# source venv/bin/activate.fish   # fish shell

# 3. instalar as dependências
pip install -r requirements.txt
```

Dependências principais: `torch`, `torchvision`, `numpy`, `matplotlib`,
`codecarbon`. Os conjuntos MNIST e Fashion-MNIST são baixados automaticamente
pelo `torchvision` na primeira execução.

---

## Reprodução dos experimentos

### Regimes MLP (MNIST e Fashion-MNIST)

O fluxo é idêntico para os dois conjuntos; basta trocar `--dataset`. Cada bloco
abaixo: (1) sintoniza os ganhos do Adam-PID e o *baseline* Adam, confirma as
melhores configurações com 5 sementes e testa o termo D; (2) sintoniza as duas
formas de SGD-PID; (3) avalia as melhores configurações de todos os otimizadores
no conjunto de teste, com medição de energia e CO₂; (4) gera as figuras.

```bash
# ---- MNIST ----
python tune_mnist.py --phase all    --dataset mnist
python tune_mnist.py --phase sgdpid --dataset mnist
python tune_mnist.py --phase final  --dataset mnist
python make_figs_mlp.py --dir experiments/tune_mnist --dataset mnist

# ---- Fashion-MNIST ----
python tune_mnist.py --phase all    --dataset fashion
python tune_mnist.py --phase sgdpid --dataset fashion
python tune_mnist.py --phase final  --dataset fashion
python make_figs_mlp.py --dir experiments/tune_fashion --dataset fashion
```

Os resultados são salvos em `experiments/tune_<dataset>/` (arquivos JSON) e as
figuras em `imagens/fase2/`. A medição de energia é ativada por padrão na fase
`final`; use `--no-energy` para desativá-la.

**Protocolo.** A sintonização ocorre **apenas no conjunto de validação** (10 000
amostras separadas do treino); o conjunto de teste permanece intocado até a fase
`final`. Todos os otimizadores partem da mesma inicialização e processam os
mini-lotes na mesma ordem dentro de cada execução.

#### Fases disponíveis (`--phase`)

| Fase      | O que faz                                                        |
|-----------|------------------------------------------------------------------|
| `grid`    | Grade completa do Adam-PID (1 semente) + *baseline* Adam         |
| `confirm` | Reavalia as melhores configurações com 5 sementes               |
| `dterm`   | Teste isolado do termo derivativo (varia *Kd*, incluindo zero)  |
| `sgdpid`  | Compara as duas formas de SGD-PID (separada e clássica)          |
| `final`   | Avalia as melhores configurações no teste, com energia e CO₂    |
| `all`     | Executa `grid` → `confirm` → `dterm` em sequência                |

### Regime CNN (CIFAR-10)

Os resultados do regime convolucional já estão versionados em `results.json` e
`confirm_seeds.json`. Para regenerá-los (requer o arquivo `cifar10.npz`, não
versionado por seu tamanho):

```bash
python run_experiment.py SGD Adam SGD-PID Adam-PID   # gera results.json com energia/CO₂
python make_figs.py                                  # curvas e convergência
python make_efficiency.py                            # tabela e figura de eficiência
```

---

## Medição de energia e emissões

O consumo elétrico é estimado com a biblioteca **[CodeCarbon](https://github.com/mlco2/codecarbon)**,
que amostra a potência da CPU e da GPU (via NVML) ao longo do treinamento e
integra-a no tempo para obter a energia em kWh. As emissões de CO₂ são
calculadas multiplicando-se a energia pelo **fator de emissão da rede brasileira**
(≈ 0,12 kg CO₂/kWh), muito abaixo da média mundial (≈ 0,475) pela forte
participação hidrelétrica.

A métrica de eficiência é o custo **até atingir a acurácia-alvo**, não o custo
total. Como as medições são estimativas de software em *hardware* de baixa
potência, reportam-se ordens de grandeza e tendências relativas, não valores
absolutos de alta precisão.

---

## Referências

1. W. An *et al.* "A PID Controller Approach for Stochastic Optimization of Deep
   Networks." *CVPR*, 2018. (Código de referência do SGD-PID:
   [tensorboy/PIDOptimizer](https://github.com/tensorboy/PIDOptimizer))
2. M. Dai *et al.* "PID controller-based adaptive gradient optimizer for deep
   neural networks." *IET Control Theory & Applications*, 2023.
3. D. P. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization." *ICLR*, 2015.
4. Y. LeCun *et al.* "Gradient-based learning applied to document recognition."
   *Proc. IEEE*, 1998. (MNIST)
5. H. Xiao *et al.* "Fashion-MNIST." *arXiv:1708.07747*, 2017.
6. A. Krizhevsky. "Learning Multiple Layers of Features from Tiny Images." 2009. (CIFAR-10)

---

## Licença e citação

Projeto acadêmico desenvolvido no CEFET-MG. Ao reutilizar o código ou os
resultados, por favor cite este repositório e os trabalhos de referência acima.

"""
make_figs_mlp.py
Gera as figuras comparativas do regime MLP a partir dos JSON de um dataset
(produzidos por tune_mnist.py). Quatro figuras:

  1. curvas de aprendizado (perda e acuracia de TESTE por epoca, media +/- banda)
  2. barras de acuracia final de teste com barra de erro (desvio entre seeds)
  3. teste isolado do termo D (acuracia vs Kd, com banda)
  4. gasto de energia (Wh por execucao e Wh ate o alvo)

Uso:
    python make_figs_mlp.py --dir experiments/tune_fashion --dataset fashion
    python make_figs_mlp.py --dir experiments/tune_mnist   --dataset mnist

Figuras salvas em --out (default: imagens/fase2).
Requires: numpy, matplotlib.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NICE = {'mnist': 'MNIST', 'fashion': 'Fashion-MNIST', 'emnist': 'EMNIST'}
COLORS = {'Adam': '#ff7f0e', 'Adam-PID': '#d62728', 'SGD': '#1f77b4',
          'SGD-mom': '#17becf', 'SGD-PID (sep)': '#2ca02c',
          'SGD-PID (clás.)': '#9467bd'}


def short_label(params):
    f = params.get('form')
    if f == 'adam_pid': return 'Adam-PID'
    if f == 'adam': return 'Adam'
    if f == 'sgd': return 'SGD' if params.get('m', 0) == 0 else 'SGD-mom'
    if f == 'sgd_sep': return 'SGD-PID (sep)'
    if f == 'sgd_classic': return 'SGD-PID (clás.)'
    return str(f)


def color_for(label):
    return COLORS.get(label, None)


def load(dirpath, name):
    with open(os.path.join(dirpath, name)) as f:
        return json.load(f)


# ----------------------------- 1. learning curves -------------------------- #
def fig_curves(final, dataset, out):
    rows = [r for r in final if not r['diverged'] and r.get('acc_curve')]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for r in rows:
        lab = short_label(r['params'])
        c = color_for(lab)
        for j, key in enumerate(('loss_curve', 'acc_curve')):
            cur = r.get(key)
            if not cur:
                continue
            m = np.array(cur['mean']); s = np.array(cur['std'])
            x = np.arange(1, len(m) + 1)
            ax[j].plot(x, m, label=lab, color=c, linewidth=1.8)
            ax[j].fill_between(x, m - s, m + s, alpha=0.18, color=c)
    ax[0].set_title('Perda de teste'); ax[0].set_xlabel('Época'); ax[0].set_ylabel('Perda')
    ax[1].set_title('Acurácia de teste'); ax[1].set_xlabel('Época'); ax[1].set_ylabel('Acurácia (%)')
    for a in ax:
        a.grid(True, alpha=0.3)
    ax[1].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Curvas de aprendizado — {NICE.get(dataset, dataset)}', fontsize=12)
    fig.tight_layout()
    p = os.path.join(out, f'fig_curvas_{dataset}.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ----------------------------- 2. accuracy bars ---------------------------- #
def fig_acc_bars(final, dataset, out):
    rows = [r for r in final if not r['diverged']]
    rows = sorted(rows, key=lambda r: r['val_mean'], reverse=True)
    labels = [short_label(r['params']) for r in rows]
    means = [r['val_mean'] for r in rows]
    stds = [r['val_std'] for r in rows]
    cols = [color_for(l) for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(rows))
    ax.bar(xs, means, yerr=stds, capsize=4, color=cols, alpha=0.9)
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.05, f'{m:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Acurácia de teste (%)')
    lo = min(means) - 2 * max(stds) - 0.5
    ax.set_ylim(lo, max(means) + 2 * max(stds) + 0.8)
    ax.set_title(f'Acurácia final (média ± desvio entre sementes) — {NICE.get(dataset, dataset)}',
                 fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, f'fig_acc_{dataset}.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ----------------------------- 3. D-term test ------------------------------ #
def fig_dterm(dirpath, dataset, out):
    try:
        d = load(dirpath, 'tune_dterm.json')
    except Exception:
        return None
    res = [r for r in d['results'] if not r['diverged']]
    res = sorted(res, key=lambda r: r['params']['Kd'])
    kd = [r['params']['Kd'] for r in res]
    m = [r['val_mean'] for r in res]
    s = [r['val_std'] for r in res]
    fig, ax = plt.subplots(figsize=(6, 4))
    m, s = np.array(m), np.array(s)
    ax.plot(kd, m, 'o-', color='#d62728', linewidth=1.8)
    ax.fill_between(kd, m - s, m + s, alpha=0.18, color='#d62728')
    ax.set_xlabel('Ganho derivativo $K_d$'); ax.set_ylabel('Acurácia de validação (%)')
    ax.set_title(f'Efeito isolado do termo D (Kp={d["best_kp"]}, Ki={d["best_ki"]}) — '
                 f'{NICE.get(dataset, dataset)}', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, f'fig_dterm_{dataset}.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ----------------------------- 4. energy bars ------------------------------ #
def fig_energy(final, dataset, out):
    rows = [r for r in final if not r['diverged'] and 'energy_wh_per_run' in r]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r['energy_wh_per_run'])
    labels = [short_label(r['params']) for r in rows]
    cols = [color_for(l) for l in labels]
    per_run = [r['energy_wh_per_run'] for r in rows]
    to_tgt = [r.get('energy_wh_to_target') for r in rows]
    has_tgt = all(t is not None for t in to_tgt)

    n = 2 if has_tgt else 1
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    xs = np.arange(len(rows))

    ax[0][0].bar(xs, per_run, color=cols, alpha=0.9)
    for x, v in zip(xs, per_run):
        ax[0][0].text(x, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax[0][0].set_xticks(xs); ax[0][0].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax[0][0].set_ylabel('Energia por execução (Wh)')
    ax[0][0].set_title('Energia total por execução', fontsize=10)
    ax[0][0].grid(axis='y', alpha=0.3)

    if has_tgt:
        ax[0][1].bar(xs, to_tgt, color=cols, alpha=0.9)
        for x, v in zip(xs, to_tgt):
            ax[0][1].text(x, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax[0][1].set_xticks(xs); ax[0][1].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax[0][1].set_ylabel('Energia até o alvo (Wh)')
        ax[0][1].set_title('Energia até a acurácia-alvo', fontsize=10)
        ax[0][1].grid(axis='y', alpha=0.3)

    fig.suptitle(f'Custo energético (rede BR) — {NICE.get(dataset, dataset)}', fontsize=12)
    fig.tight_layout()
    p = os.path.join(out, f'fig_energia_{dataset}.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='pasta com os JSON (ex: experiments/tune_fashion)')
    ap.add_argument('--dataset', required=True, choices=list(NICE))
    ap.add_argument('--out', default='imagens/fase2')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    final = load(args.dir, 'final_results.json')['final_test']
    made = []
    made.append(fig_curves(final, args.dataset, args.out))
    made.append(fig_acc_bars(final, args.dataset, args.out))
    made.append(fig_dterm(args.dir, args.dataset, args.out))
    made.append(fig_energy(final, args.dataset, args.out))
    for p in made:
        print('salvo ->', p) if p else print('(figura pulada: dados ausentes)')


if __name__ == '__main__':
    main()

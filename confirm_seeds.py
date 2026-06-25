"""
confirm_seeds.py
Confirma as duas configs FINALISTAS com varias seeds, na VALIDACAO:
  - Adam de base:  lr=0.003                       (baseline justo, lr afinada)
  - Adam-PID:      Kp=1.0, Ki=1, Kd=0.3, lr=1e-3  (melhor da busca)

Para cada config, reporta:
  - acuracia final: media +/- desvio entre seeds  (empate ou nao no teto)
  - epocas-ate-o-alvo: media +/- desvio           (VELOCIDADE de convergencia)

A segunda metrica e' a alegacao real dos artigos de PID: mesmo que todos
saturem no mesmo plato, um metodo pode chegar la com menos epocas.

Obs. importante: a seed aqui varia a INICIALIZACAO da rede. A ordem dos
mini-batches em conv.train e' fixa (rng 123). Se quiser que a seed varie
TAMBEM a ordem dos batches (variancia de run-a-run mais completa), troque em
conv.py a linha  rng = np.random.default_rng(123)  por  (123 + seed).
"""
import json
import numpy as np

import core
import conv
from tune_pid import load_data


def epochs_to(h, thr):
    """Primeira epoca cuja acuracia de validacao cruza thr% (None se nunca)."""
    for e, a in zip(h["epoch"], h["test_acc"]):
        if a * 100 >= thr:
            return e
    return None


def run_seeds(data, configs, seeds, target=45.0, epochs=8, batch=128, max_train=8000):
    x_tr, y_tr, x_val, y_val = data[0], data[1], data[2], data[3]
    summary = {}
    for name, make in configs.items():
        accs, eps = [], []
        for s in seeds:
            h = conv.train(make, (x_tr, y_tr, x_val, y_val),
                           epochs=epochs, batch=batch, seed=s, max_train=max_train)
            acc = h["test_acc"][-1] * 100
            e = epochs_to(h, target)
            accs.append(acc)
            eps.append(e)
            print(f"  {name:26s} seed {s}: val {acc:5.2f}%  "
                  f"->{target:.0f}% em {e if e else '—'} ep", flush=True)
        accs = np.array(accs)
        reached = [e for e in eps if e is not None]
        summary[name] = {
            "accs": accs.tolist(),
            "val_mean": float(accs.mean()),
            "val_std": float(accs.std(ddof=1)),
            "ep_to_target": eps,
            "ep_mean": (float(np.mean(reached)) if reached else None),
            "ep_std": (float(np.std(reached, ddof=1)) if len(reached) > 1 else None),
            "n_reached": len(reached),
        }
        print()
    return summary


def report(summary, seeds, target):
    print("=== RESUMO (validacao) ===")
    head = f"{'Config':26s} {'AccFinal (media±dp)':>22s} {'Ep.→alvo (media±dp)':>22s}"
    print(head)
    print("-" * len(head))
    for name, s in summary.items():
        acc = f"{s['val_mean']:.2f} ± {s['val_std']:.2f}%"
        if s["ep_mean"] is not None:
            dp = s["ep_std"] if s["ep_std"] is not None else 0.0
            ep = f"{s['ep_mean']:.1f} ± {dp:.1f}"
            if s["n_reached"] < len(seeds):
                ep += f" ({s['n_reached']}/{len(seeds)})"
        else:
            ep = "nao atingiu"
        print(f"{name:26s} {acc:>22s} {ep:>22s}")

    names = list(summary)
    if len(names) == 2:
        a, b = summary[names[0]], summary[names[1]]
        diff = abs(a["val_mean"] - b["val_mean"])
        spread = a["val_std"] + b["val_std"]
        print(f"\nDiferenca de acuracia final: {diff:.2f}%  "
              f"(soma dos desvios: {spread:.2f}%)")
        if diff <= spread:
            print("-> DENTRO da variacao entre seeds: empate estatistico "
                  "(nao e' um teste formal, mas e' o sinal de alerta certo).")
        else:
            print("-> SUPERA a variacao entre seeds: possivel efeito real "
                  "(confirme so entao no conjunto de teste).")


if __name__ == "__main__":
    SEEDS = [0, 1, 2, 3, 4]
    TARGET = 45.0      # alvo (%) abaixo do plato (~53%), p/ medir convergencia

    configs = {
        "Adam (lr=0.003)":
            lambda p: core.Adam(p, lr=3e-3),
        "Adam-PID (Kp1.0,Kd0.3)":
            lambda p: core.Adam_PID(p, lr=1e-3, gamma=0.9, Kp=1.0, Ki=1.0, Kd=0.3),
    }

    data = load_data("cifar10.npz")
    summary = run_seeds(data, configs, SEEDS, target=TARGET)
    report(summary, SEEDS, TARGET)
    json.dump(summary, open("confirm_seeds.json", "w"), indent=2)
    print("\nsalvo -> confirm_seeds.json")

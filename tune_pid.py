"""
tune_pid.py
Busca em grade dos ganhos Kp, Ki, Kd do Adam_PID, usando o conjunto de
VALIDACAO (nunca o de teste).

Estrategia:
  - fixa Ki=1 (mantem a parte integral = Adam) e varre Kp e Kd -> grade 2D barata;
  - cada tentativa usa orcamento REDUZIDO (max_train, poucas epocas) p/ rapidez:
    o RANKING entre configs costuma se preservar em escala menor, e ai voce
    confirma so as melhores em escala cheia;
  - configs que divergem (NaN/inf) sao descartadas automaticamente;
  - imprime o ranking por acuracia de validacao e salva tune_results.json.

Truque: passamos a VALIDACAO no lugar do "teste" para conv.train, entao
hist["test_acc"] ja e' a acuracia de validacao -- sem mexer no laco de treino.
"""
import json
import itertools
import numpy as np

import core
import conv


def load_data(path="cifar10.npz"):
    d = np.load(path)
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    prep = lambda x: (x.astype(np.float32) / 255.0 - mean) / std
    return (prep(d["x_train"]), d["y_train"],
            prep(d["x_val"]),   d["y_val"],
            prep(d["x_test"]),  d["y_test"])


def trial(x_tr, y_tr, x_val, y_val, Kp, Ki, Kd,
          lr=1e-3, gamma=0.9, epochs=8, batch=128, seed=0, max_train=8000):
    """Treina uma config e devolve a acuracia de validacao (None se divergir)."""
    make = lambda p: core.Adam_PID(p, lr=lr, gamma=gamma, Kp=Kp, Ki=Ki, Kd=Kd)
    h = conv.train(make, (x_tr, y_tr, x_val, y_val),
                   epochs=epochs, batch=batch, seed=seed, max_train=max_train)
    val = h["test_acc"][-1]                      # = acuracia de validacao
    if not np.isfinite(val) or not np.isfinite(h["train_loss"][-1]):
        return None
    return {"Kp": Kp, "Ki": Ki, "Kd": Kd, "val_acc": float(val),
            "val_curve": [round(float(a), 4) for a in h["test_acc"]]}


def run_grid(data, kp_grid, kd_grid, ki=1.0, **kw):
    x_tr, y_tr, x_val, y_val = data[0], data[1], data[2], data[3]
    results = []
    for Kp, Kd in itertools.product(kp_grid, kd_grid):
        r = trial(x_tr, y_tr, x_val, y_val, Kp, ki, Kd, **kw)
        tag = "DIVERGIU" if r is None else f"val {r['val_acc'] * 100:5.2f}%"
        print(f"Kp={Kp:<5} Ki={ki:<4} Kd={Kd:<5} -> {tag}", flush=True)
        if r is not None:
            results.append(r)
    results.sort(key=lambda r: r["val_acc"], reverse=True)
    return results


if __name__ == "__main__":
    data = load_data("cifar10.npz")

    # Grade GROSSA (Ki fixo = 1; varre Kp e Kd). Refine depois ao redor do melhor.
    KP_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
    KD_GRID = [0.0, 0.3]

    results = run_grid(data, KP_GRID, KD_GRID, ki=1.0,
                       lr=1e-3, epochs=8, batch=128, seed=0, max_train=8000)
    
    # --- linha de base JUSTA: Adam puro (Kp=0, Kd=0) com varias lr ---
    print("\n=== Adam de base (Kp=0, Kd=0) variando lr ===")
    base = []
    for lr in [1e-3, 2e-3, 3e-3, 5e-3]:
        r = trial(data[0], data[1], data[2], data[3], 0.0, 1.0, 0.0,
                  lr=lr, epochs=8, batch=128, seed=0, max_train=8000)
        if r is not None:
            r["lr"] = lr
            base.append(r)
        tag = "DIVERGIU" if r is None else f"val {r['val_acc'] * 100:5.2f}%"
        print(f"lr={lr:<7} -> {tag}", flush=True)
    base.sort(key=lambda r: r["val_acc"], reverse=True)
    if base:
        b0 = base[0]
        print(f"Melhor Adam de base: lr={b0['lr']}, val {b0['val_acc'] * 100:.2f}%")

    print("\n=== TOP 5 (validacao) ===")
    for r in results[:5]:
        print(f"  Kp={r['Kp']:<5} Ki={r['Ki']:<4} Kd={r['Kd']:<5} "
              f"val {r['val_acc'] * 100:.2f}%")
    json.dump(results, open("tune_results.json", "w"), indent=2)
    print("\nsalvo -> tune_results.json")
    if results:
        b = results[0]
        print(f"\nMelhor (grade grossa): Kp={b['Kp']}, Ki={b['Ki']}, Kd={b['Kd']}.")
        print("Proximo: refine numa grade fina ao redor desses valores e, com a")
        print("config vencedora, rode VARIAS seeds antes de tocar no conjunto de teste.")

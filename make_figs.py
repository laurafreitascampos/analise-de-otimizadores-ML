"""Gera as figuras e a tabela-resumo a partir de results.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("results.json"))
order = ["SGD", "Adam", "SGD-PID", "Adam-PID"]
colors = {"SGD": "#1f77b4", "Adam": "#ff7f0e", "SGD-PID": "#2ca02c", "Adam-PID": "#d62728"}
styles = {"SGD": "-", "Adam": "-", "SGD-PID": "--", "Adam-PID": "--"}

# ---------- Figura 1: 4 paineis ----------
fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
panels = [("train_loss", "Perda de treino", ax[0, 0]),
          ("test_loss",  "Perda de teste",  ax[0, 1]),
          ("train_acc",  "Acuracia de treino (%)", ax[1, 0]),
          ("test_acc",   "Acuracia de teste (%)",  ax[1, 1])]
for key, title, a in panels:
    for n in order:
        y = r[n][key]
        if "acc" in key:
            y = [v * 100 for v in y]
        a.plot(r[n]["epoch"], y, styles[n], color=colors[n], label=n, linewidth=1.8)
    a.set_title(title, fontsize=10)
    a.set_xlabel("Epoca", fontsize=9)
    a.grid(True, alpha=0.3)
    a.tick_params(labelsize=8)
ax[0, 0].legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("fig_curvas.png", dpi=150)
print("fig_curvas.png salvo")

# ---------- Figura 2: epocas para atingir limiar ----------
def epochs_to(n, thr):
    for e, acc in zip(r[n]["epoch"], r[n]["test_acc"]):
        if acc * 100 >= thr:
            return e
    return None

thr = 98.0
vals = [epochs_to(n, thr) for n in order]
plt.figure(figsize=(5.2, 3.4))
xs = np.arange(len(order))
bars = plt.bar(xs, [v if v else 0 for v in vals],
               color=[colors[n] for n in order])
for x, v in zip(xs, vals):
    plt.text(x, (v if v else 0) + 0.05, str(v) if v else "—",
             ha="center", va="bottom", fontsize=9)
plt.xticks(xs, order, fontsize=9)
plt.ylabel(f"Epocas ate atingir {thr:.0f}% (teste)", fontsize=9)
plt.title("Velocidade de convergencia (menor = melhor)", fontsize=10)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_convergencia.png", dpi=150)
print("fig_convergencia.png salvo")

# ---------- Tabela-resumo ----------
print("\n=== RESUMO ===")
print(f"{'Otimizador':<10} {'AccFinal':>9} {'AccMax':>8} {'->97%':>6} {'->98%':>6} "
      f"{'PerdaTrFinal':>13} {'Tempo(s)':>9}")
for n in order:
    h = r[n]
    acc_final = h["test_acc"][-1] * 100
    acc_max = max(h["test_acc"]) * 100
    e97 = epochs_to(n, 97.0); e98 = epochs_to(n, 98.0)
    tempo = h["cum_time"][-1] if "cum_time" in h else float("nan")
    print(f"{n:<10} {acc_final:>8.2f}% {acc_max:>7.2f}% "
          f"{(str(e97) if e97 else '—'):>6} {(str(e98) if e98 else '—'):>6} "
          f"{h['train_loss'][-1]:>13.4f} {tempo:>9.1f}")
print("\nmeta:", r["_meta"])

"""
Calcula metricas de EFICIENCIA a partir de results.json:
  - tempo de computo ate atingir a acuracia-alvo (s)
  - energia ate o alvo (Wh)  [proporcional ao tempo]
  - CO2 ate o alvo (g, fator do Brasil)
  - FLOPs ate o alvo
Gera fig_eficiencia.png e imprime a tabela.

Ideia: a energia total favorece quem roda menos epocas; o que importa para
eficiencia e o CUSTO PARA CHEGAR A UM MESMO RESULTADO. Por isso medimos tudo
ate o instante em que cada otimizador cruza a acuracia-alvo.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("results.json"))
order = ["SGD", "Adam", "SGD-PID", "Adam-PID"]
colors = {"SGD": "#1f77b4", "Adam": "#ff7f0e", "SGD-PID": "#2ca02c", "Adam-PID": "#d62728"}
TARGET = 97.0   # acuracia-alvo (%) atingivel por TODOS os otimizadores


def idx_to_target(h, thr):
    """indice da primeira epoca que atinge thr% de acuracia de teste."""
    for i, acc in enumerate(h["test_acc"]):
        if acc * 100 >= thr:
            return i
    return None


def metrics_to_target(h, thr):
    i = idx_to_target(h, thr)
    if i is None:
        return None
    # fracao de energia/tempo total proporcional a epoca atingida
    ep_frac = h["epoch"][i] / h["epoch"][-1]
    time_total = h["cum_time"][i]
    flops = h["cum_flops"][i]
    out = {"epoch": h["epoch"][i], "time_s": time_total, "flops": flops}
    if "energy_kwh" in h:
        # energia ate o alvo ~ energia_total * (tempo_ate_alvo / tempo_total)
        frac = h["cum_time"][i] / h["cum_time"][-1]
        out["energy_wh"] = h["energy_kwh"] * 1000.0 * frac
        out["co2_g"] = h["co2_g_brasil"] * frac
    return out


print(f"=== EFICIENCIA ATE {TARGET:.1f}% DE ACURACIA DE TESTE ===\n")
hdr = f"{'Otimizador':<10} {'Epoca':>6} {'Tempo(s)':>9} {'GFLOPs':>9}"
has_energy = "energy_kwh" in r["SGD"]
if has_energy:
    hdr += f" {'Energia(Wh)':>12} {'CO2(g)':>8}"
print(hdr)
print("-" * len(hdr))

rows = {}
for n in order:
    m = metrics_to_target(r[n], TARGET)
    rows[n] = m
    if m is None:
        print(f"{n:<10} {'nao atinge o alvo em '+str(r[n]['epoch'][-1])+' epocas':>40}")
        continue
    line = f"{n:<10} {m['epoch']:>6} {m['time_s']:>9.1f} {m['flops']/1e9:>9.2f}"
    if has_energy:
        line += f" {m['energy_wh']:>12.3f} {m['co2_g']:>8.4f}"
    print(line)

# tabela tambem com o TOTAL das 10 epocas (custo absoluto)
print(f"\n=== CUSTO TOTAL ({r['_meta']['epochs']} epocas completas) ===\n")
hdr2 = f"{'Otimizador':<10} {'Tempo(s)':>9} {'GFLOPs':>9}"
if has_energy:
    hdr2 += f" {'Energia(Wh)':>12} {'CO2(g)':>8} {'Pot.med(W)':>11}"
print(hdr2)
print("-" * len(hdr2))
for n in order:
    h = r[n]
    line = f"{n:<10} {h['cum_time'][-1]:>9.1f} {h['cum_flops'][-1]/1e9:>9.2f}"
    if has_energy:
        line += (f" {h['energy_kwh']*1000:>12.3f} {h['co2_g_brasil']:>8.4f}"
                 f" {h.get('avg_power_w',0):>11.1f}")
    print(line)

# ---------- Figura de eficiencia ----------
if has_energy and all(rows[n] for n in order):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    xs = np.arange(len(order))

    t_vals = [rows[n]["time_s"] for n in order]
    ax[0].bar(xs, t_vals, color=[colors[n] for n in order])
    for x, v in zip(xs, t_vals):
        ax[0].text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(order, fontsize=8, rotation=15)
    ax[0].set_ylabel("Tempo de computo (s)", fontsize=9)
    ax[0].set_title(f"Tempo ate {TARGET:.1f}% de acuracia", fontsize=10)
    ax[0].grid(axis="y", alpha=0.3)

    e_vals = [rows[n]["energy_wh"] for n in order]
    ax[1].bar(xs, e_vals, color=[colors[n] for n in order])
    for x, v in zip(xs, e_vals):
        ax[1].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(order, fontsize=8, rotation=15)
    ax[1].set_ylabel("Energia estimada (Wh)", fontsize=9)
    ax[1].set_title(f"Energia ate {TARGET:.1f}% (rede BR)", fontsize=10)
    ax[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("fig_eficiencia.png", dpi=150)
    print("\nfig_eficiencia.png salvo")

"""
Roda otimizadores (passados na linha de comando) e ACUMULA em results.json.
Registra, alem das metricas de aprendizado: tempo de computo, FLOPs e uma
estimativa de energia/CO2 (via codecarbon, com fator de emissao do Brasil).
"""
import sys, os, json, time
import numpy as np
import core
import conv  # além de core

# Fator de emissao da rede eletrica BRASILEIRA (matriz limpa, muita hidro).
# ~0,12 kg CO2 / kWh. (Media mundial fica perto de 0,475.) Ajuste se desejar.
BRASIL_KG_CO2_POR_KWH = 0.12

try:
    from codecarbon import EmissionsTracker
    HAS_CC = True
except Exception:
    HAS_CC = False

d = np.load("cifar10.npz")
mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
std  = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
def prep(x): return (x.astype(np.float32) / 255.0 - mean) / std
data = (prep(d["x_train"]), d["y_train"], prep(d["x_test"]), d["y_test"])

EPOCHS, BATCH = 10, 128


cfgs = {
    "SGD":      lambda p: core.SGD(p, lr=0.1),
    "Adam":     lambda p: core.Adam(p, lr=1e-3),
    "SGD-PID":  lambda p: core.SGD_PID(p, lr=0.1, alpha=0.9, Kd=0.5),
    "Adam-PID": lambda p: core.Adam_PID(p, lr=1e-3, gamma=0.9, Kp=0.8, Ki=1.0, Kd=0.5),
}

results = json.load(open("results.json")) if os.path.exists("results.json") else {}
names = sys.argv[1:]
t0 = time.time()
for name in names:
    tracker = None
    if HAS_CC:
        tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False,
                                   log_level="error")
        tracker.start()

    h = conv.train(cfgs[name], data, epochs=EPOCHS, batch=BATCH, seed=0, max_train=None)

    if tracker is not None:
        tracker.stop()
        fed = tracker.final_emissions_data
        energy_kwh = float(fed.energy_consumed)          # kWh
        avg_power_w = float(fed.cpu_power or 0.0)         # W
        h["energy_kwh"] = energy_kwh
        h["avg_power_w"] = avg_power_w
        # CO2 recalculado com o fator do Brasil (codecarbon usa o pais do IP)
        h["co2_g_brasil"] = energy_kwh * BRASIL_KG_CO2_POR_KWH * 1000.0

    results[name] = h
    extra = ""
    if "energy_kwh" in h:
        extra = (f"  energia {h['energy_kwh']*1000:.3f} Wh  "
                 f"CO2 {h['co2_g_brasil']:.3f} g")
    print(f"[{time.time()-t0:6.1f}s] {name:9s} teAcc {h['test_acc'][-1]*100:.2f}%  "
          f"compute {h['cum_time'][-1]:.1f}s{extra}", flush=True)

results["_meta"] = {"epochs": EPOCHS, "batch": BATCH,
                    "n_train": int(data[0].shape[0]), "n_test": int(data[2].shape[0]),
                    "brasil_kg_co2_por_kwh": BRASIL_KG_CO2_POR_KWH,
                    "energia_medida": HAS_CC}
json.dump(results, open("results.json", "w"))
print("salvo. parcial total %.1fs" % (time.time()-t0))

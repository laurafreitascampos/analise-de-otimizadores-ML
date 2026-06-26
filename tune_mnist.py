"""
tune_mnist.py
Sintonizacao dos ganhos PID na MLP/MNIST, no harness rapido (GPU).

Protocolo (espelha tune_pid.py + confirm_seeds.py da Fase 1, agora em PyTorch):
  - separa VALIDACAO do treino (50k treino / 10k validacao); o TESTE fica
    intocado ate a configuracao final;
  - estrategia em duas etapas: grade completa com 1 semente para achar as
    finalistas, depois 5 sementes so nas melhores (media +/- desvio);
  - configs que divergem (NaN/Inf) sao descartadas;
  - tudo salvo em JSON para virar tabela/figura do artigo.

Fases (--phase):
  grid    : grade completa Adam-PID (1 seed) + baseline Adam (sweep de lr)
  confirm : top-K da grade com 5 seeds (val media +/- desvio)
  dterm   : teste isolado do termo D (fixa melhor Kp,Ki; varia Kd incl. 0) 5 seeds
  sgdpid  : compara as DUAS formas de SGD-PID (separada vs classica An et al.)
  all     : grid -> confirm -> dterm  (recomendado para o run principal)

Exemplo (fish):
    python tune_mnist.py --phase all
    python tune_mnist.py --phase sgdpid

Requires: torch, torchvision, numpy.  (sem scipy)
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from compare_mnist_mlp import (make_mlp, set_seed, evaluate, load_full, DATASETS,
                               MNIST_MEAN, MNIST_STD)
from pid_optimizers import SGD_PID, SGD_PID_Classic, Adam_PID

# Fator de emissao da rede eletrica brasileira (matriz limpa, muita hidro).
BRASIL_KG_CO2_POR_KWH = 0.12
try:
    from codecarbon import EmissionsTracker
    HAS_CC = True
except Exception:
    HAS_CC = False


# ------------------------------- data -------------------------------------- #
def load_data_with_val(dataset, data_dir, device, val_size=10000, split_seed=0):
    """Train -> train+val split; test kept separate and untouched.
    Returns (train, val, test, num_classes)."""
    (Xtr_all, Ytr_all), (Xte, Yte), num_classes = load_full(dataset, data_dir, device)

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(Xtr_all.size(0), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    Xtr, Ytr = Xtr_all[tr_idx], Ytr_all[tr_idx]
    Xval, Yval = Xtr_all[val_idx], Ytr_all[val_idx]
    return (Xtr, Ytr), (Xval, Yval), (Xte, Yte), num_classes


# ----------------------- train one config, eval on val -------------------- #
def train_eval(opt_factory, train, val, hidden, epochs, batch, seed,
               target, clip, device, num_classes=10):
    Xtr, Ytr = train
    Xval, Yval = val
    set_seed(seed)
    model = make_mlp(hidden, num_classes).to(device)
    opt = opt_factory(model.parameters())
    crit = nn.CrossEntropyLoss()
    N = Xtr.size(0)

    val_curve, loss_curve, ep_to_target, diverged = [], [], None, False
    cum, cum_time, time_to_target = 0.0, [], None
    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            out = model(Xtr[idx])
            loss = crit(out, Ytr[idx])
            if not math.isfinite(loss.item()):
                diverged = True
                break
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if diverged:
            break
        if device.type == 'cuda':
            torch.cuda.synchronize()
        cum += time.perf_counter() - t0          # training compute only (excl. eval)
        cum_time.append(cum)
        vl, va = evaluate(model, Xval, Yval, crit)
        val_curve.append(va)
        loss_curve.append(vl)
        if ep_to_target is None and va >= target:
            ep_to_target = epoch + 1
            time_to_target = cum

    return {'val_acc': (val_curve[-1] if val_curve else float('nan')),
            'val_curve': val_curve, 'loss_curve': loss_curve,
            'ep_to_target': ep_to_target, 'diverged': diverged,
            'total_time': (cum_time[-1] if cum_time else 0.0),
            'time_to_target': time_to_target}


# -------- run a list of configs over given seeds, aggregate val acc -------- #
def run_configs(configs, data, args, seeds, label, eval_split='val'):
    """configs: list of (tag, factory, params). Returns ranked list of dicts.
    eval_split='val' evaluates on the validation set (tuning); 'test' evaluates
    on the untouched test set (final reporting). Note: in the returned dict the
    accuracy key is 'val_mean' regardless of split, so when eval_split='test'
    it actually holds the TEST accuracy."""
    train = data[0]
    val = data[1] if eval_split == 'val' else data[2]
    out = []
    for tag, factory, params in configs:
        accs, eps, div = [], [], 0
        for s in seeds:
            r = train_eval(factory, train, val, args.hidden, args.epochs,
                           args.batch, s, args.target, args.clip, args.device,
                           args.num_classes)
            if r['diverged'] or not math.isfinite(r['val_acc']):
                div += 1
                continue
            accs.append(r['val_acc'] * 100)
            eps.append(r['ep_to_target'])
        if not accs:
            print(f"  [{label}] {tag:38s} -> DIVERGIU (todas as {len(seeds)} seeds)")
            out.append({'tag': tag, 'params': params, 'diverged': True})
            continue
        a = np.array(accs)
        reached = [e for e in eps if e is not None]
        rec = {'tag': tag, 'params': params, 'diverged': False,
               'n_seeds': len(seeds), 'n_div': div, 'accs': accs,
               'val_mean': float(a.mean()),
               'val_std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
               'ep_to_target': eps,
               'ep_mean': float(np.mean(reached)) if reached else None,
               'ep_std': float(np.std(reached, ddof=1)) if len(reached) > 1 else None,
               'n_reached': len(reached)}
        std_str = f"+/- {rec['val_std']:.2f}" if len(seeds) > 1 else ""
        print(f"  [{label}] {tag:38s} -> val {rec['val_mean']:5.2f} {std_str}", flush=True)
        out.append(rec)
    ok = [r for r in out if not r['diverged']]
    ok.sort(key=lambda r: r['val_mean'], reverse=True)
    return ok + [r for r in out if r['diverged']]


# ------------------------------ factories ---------------------------------- #
# Each returns (tag, factory, params). params is carried through run_configs so
# we never re-parse tags (which was fragile).
def f_adam_pid(Kp, Ki, Kd, lr):
    return (f"AdamPID Kp{Kp} Ki{Ki} Kd{Kd}",
            lambda p: Adam_PID(p, lr=lr, Kp=Kp, Ki=Ki, Kd=Kd, weight_decay=0.0),
            {'form': 'adam_pid', 'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'lr': lr})

def f_adam(lr):
    return (f"Adam lr{lr}",
            lambda p: torch.optim.Adam(p, lr=lr, weight_decay=0.0),
            {'form': 'adam', 'lr': lr})

def f_sgd_pid_sep(Kp, Ki, Kd, lr, m):
    return (f"SGDPID-sep Kp{Kp} Ki{Ki} Kd{Kd}",
            lambda p: SGD_PID(p, lr=lr, momentum=m, Kp=Kp, Ki=Ki, Kd=Kd),
            {'form': 'sgd_sep', 'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'lr': lr, 'm': m})

def f_sgd_pid_classic(alpha, Kd, lr):
    return (f"SGDPID-classic a{alpha} Kd{Kd} lr{lr}",
            lambda p: SGD_PID_Classic(p, lr=lr, alpha=alpha, Kd=Kd),
            {'form': 'sgd_classic', 'alpha': alpha, 'Kd': Kd, 'lr': lr})


def rebuild(params):
    """Reconstruct a (tag, factory, params) triple from a stored params dict."""
    f = params['form']
    if f == 'adam_pid':
        return f_adam_pid(params['Kp'], params['Ki'], params['Kd'], params['lr'])
    if f == 'adam':
        return f_adam(params['lr'])
    if f == 'sgd':
        return (f"SGD lr{params['lr']} m{params['m']}",
                lambda p: torch.optim.SGD(p, lr=params['lr'], momentum=params['m']),
                params)
    if f == 'sgd_sep':
        return f_sgd_pid_sep(params['Kp'], params['Ki'], params['Kd'], params['lr'], params['m'])
    if f == 'sgd_classic':
        return f_sgd_pid_classic(params['alpha'], params['Kd'], params['lr'])
    raise ValueError(f"unknown form {f}")


def save(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"  salvo -> {path}")


# ------------------------------- phases ------------------------------------ #
def phase_grid(data, args):
    print("\n=== FASE GRID: Adam-PID grade completa (1 seed) ===")
    grid = [f_adam_pid(kp, ki, kd, args.adam_lr)
            for kp in args.kp_grid for ki in args.ki_grid for kd in args.kd_grid]
    ranked = run_configs(grid, data, args, seeds=[0], label="grid")

    print("\n=== Baseline Adam puro (sweep de lr, 1 seed) ===")
    base = run_configs([f_adam(lr) for lr in args.adam_lr_grid],
                       data, args, seeds=[0], label="base")

    save({'adam_pid_ranked': ranked, 'adam_baseline': base},
         os.path.join(args.out_dir, 'tune_grid.json'))
    print("\nTOP 5 Adam-PID (val, 1 seed):")
    for r in [x for x in ranked if not x['diverged']][:5]:
        print(f"  {r['tag']:38s} val {r['val_mean']:.2f}")
    return ranked, base


def phase_confirm(data, args, ranked=None):
    print(f"\n=== FASE CONFIRM: top-{args.topk} com {len(args.seeds)} seeds ===")
    if ranked is None:
        ranked = json.load(open(os.path.join(args.out_dir, 'tune_grid.json')))['adam_pid_ranked']
    top = [r for r in ranked if not r['diverged']][:args.topk]
    configs = [rebuild(r['params']) for r in top]
    confirmed = run_configs(configs, data, args, seeds=args.seeds, label="confirm")
    save({'confirmed': confirmed},
         os.path.join(args.out_dir, 'tune_confirm.json'))
    return confirmed


def phase_dterm(data, args, best_kp=None, best_ki=None):
    print(f"\n=== FASE DTERM: efeito isolado do termo D ({len(args.seeds)} seeds) ===")
    if best_kp is None or best_ki is None:
        try:
            conf = json.load(open(os.path.join(args.out_dir, 'tune_confirm.json')))['confirmed']
            best = [c for c in conf if not c['diverged']][0]['params']
            best_kp, best_ki = best['Kp'], best['Ki']
        except Exception:
            best_kp, best_ki = args.kp_grid[len(args.kp_grid) // 2], 1.0
    print(f"  fixando Kp={best_kp}, Ki={best_ki}; variando Kd em {args.kd_grid}")
    configs = [f_adam_pid(best_kp, best_ki, kd, args.adam_lr) for kd in args.kd_grid]
    res = run_configs(configs, data, args, seeds=args.seeds, label="dterm")
    save({'best_kp': best_kp, 'best_ki': best_ki, 'kd_grid': args.kd_grid,
          'results': res}, os.path.join(args.out_dir, 'tune_dterm.json'))
    return res


def phase_sgdpid(data, args):
    print(f"\n=== FASE SGDPID: duas formas, cada uma na sua grade ===")
    print("  forma SEPARADA (tres ganhos):")
    sep = [f_sgd_pid_sep(kp, ki, kd, args.sgd_lr, args.momentum)
           for kp in [0.5, 1.0] for ki in [0.0, 0.5, 1.0] for kd in args.kd_grid]
    sep_ranked = run_configs(sep, data, args, seeds=[0], label="sgd-sep")

    print("  forma CLASSICA (An et al.: alpha, Kd, lr):")
    cls = [f_sgd_pid_classic(a, kd, lr)
           for a in [0.9] for kd in [0.0, 0.3, 0.5, 1.0] for lr in [0.01, 0.05, 0.1]]
    cls_ranked = run_configs(cls, data, args, seeds=[0], label="sgd-cls")

    # confirm the best of each form with multiple seeds (params carried, no parsing)
    confirm_cfgs = []
    for ranked in (sep_ranked, cls_ranked):
        best = [r for r in ranked if not r['diverged']]
        if best:
            confirm_cfgs.append(rebuild(best[0]['params']))
    print(f"\n  confirmando melhores de cada forma com {len(args.seeds)} seeds:")
    confirmed = run_configs(confirm_cfgs, data, args, seeds=args.seeds, label="sgd-final")
    save({'separated_ranked': sep_ranked, 'classic_ranked': cls_ranked,
          'confirmed': confirmed}, os.path.join(args.out_dir, 'tune_sgdpid.json'))
    return confirmed


def run_final_with_energy(configs, data, args, seeds):
    """Train each config on train, eval on TEST, over all seeds, and (if
    CodeCarbon is available and --energy is set) measure energy/CO2 per config.
    Energy-to-target is estimated as total energy * (time_to_target/total_time),
    consistent with make_efficiency.py."""
    train, test = data[0], data[2]
    measure = args.energy and HAS_CC

    def agg_curve(curves):
        curves = [c for c in curves if c]
        if not curves:
            return None
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        return {'mean': arr.mean(0).tolist(),
                'std': arr.std(0).tolist() if arr.shape[0] > 1 else [0.0] * n}

    out = []
    for tag, factory, params in configs:
        accs, eps, tot_times, ttt_times = [], [], [], []
        acc_curves, loss_curves = [], []
        tracker = None
        if measure:
            tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False,
                                       log_level="error")
            tracker.start()
        for s in seeds:
            r = train_eval(factory, train, test, args.hidden, args.epochs,
                           args.batch, s, args.target, args.clip, args.device,
                           args.num_classes)
            if r['diverged'] or not math.isfinite(r['val_acc']):
                continue
            accs.append(r['val_acc'] * 100)
            eps.append(r['ep_to_target'])
            tot_times.append(r['total_time'])
            ttt_times.append(r['time_to_target'])
            acc_curves.append([a * 100 for a in r['val_curve']])   # test acc (%) per epoch
            loss_curves.append(r['loss_curve'])                    # test loss per epoch
        energy_kwh = power_w = co2_g = None
        if tracker is not None:
            tracker.stop()
            fed = tracker.final_emissions_data
            energy_kwh = float(fed.energy_consumed)
            power_w = float((fed.cpu_power or 0.0) + (getattr(fed, 'gpu_power', 0.0) or 0.0))
            co2_g = energy_kwh * BRASIL_KG_CO2_POR_KWH * 1000.0

        if not accs:
            out.append({'tag': tag, 'params': params, 'diverged': True})
            continue
        a = np.array(accs)
        reached = [e for e in eps if e is not None]
        n = len(accs)
        rec = {'tag': tag, 'params': params, 'diverged': False, 'n_seeds': len(seeds),
               'accs': accs, 'val_mean': float(a.mean()),
               'val_std': float(a.std(ddof=1)) if n > 1 else 0.0,
               'ep_mean': float(np.mean(reached)) if reached else None,
               'ep_std': float(np.std(reached, ddof=1)) if len(reached) > 1 else None,
               'time_per_run_s': float(np.mean(tot_times)),
               'time_to_target_s': (float(np.nanmean([t for t in ttt_times if t is not None]))
                                    if any(t is not None for t in ttt_times) else None),
               'acc_curve': agg_curve(acc_curves),
               'loss_curve': agg_curve(loss_curves)}
        if energy_kwh is not None:
            rec['energy_kwh_total'] = energy_kwh
            rec['energy_wh_per_run'] = energy_kwh * 1000.0 / n
            rec['co2_g_total'] = co2_g
            rec['co2_g_per_run'] = co2_g / n
            rec['avg_power_w'] = power_w
            # energy/CO2 ATE o alvo ~ por-run * (tempo_ate_alvo / tempo_total)
            if rec['time_to_target_s'] and rec['time_per_run_s'] > 0:
                frac = rec['time_to_target_s'] / rec['time_per_run_s']
                rec['energy_wh_to_target'] = rec['energy_wh_per_run'] * frac
                rec['co2_g_to_target'] = rec['co2_g_per_run'] * frac
        e_str = (f" | {rec.get('energy_wh_per_run', float('nan')):.3f} Wh/run"
                 if energy_kwh is not None else "")
        print(f"  [final] {tag:38s} -> test {rec['val_mean']:5.2f} "
              f"+/- {rec['val_std']:.2f}{e_str}", flush=True)
        out.append(rec)
    out_ok = [r for r in out if not r['diverged']]
    out_ok.sort(key=lambda r: r['val_mean'], reverse=True)
    return out_ok + [r for r in out if r['diverged']]


def phase_final(data, args):
    """Best config of each optimizer, evaluated on the UNTOUCHED test set,
    same seeds, with energy/CO2, consolidated into one table/JSON."""
    print(f"\n=== FASE FINAL: melhores configs no TESTE intocado ({len(args.seeds)} seeds) ===")
    if args.energy and not HAS_CC:
        print("  (CodeCarbon nao encontrado: rodando sem energia. pip install codecarbon)")
    od = args.out_dir
    configs = []
    configs.append(("SGD (artigo) lr0.1 m0",
                    lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.0),
                    {'form': 'sgd', 'lr': 0.1, 'm': 0.0}))
    configs.append(("SGD-mom (nosso) lr0.1 m0.9",
                    lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
                    {'form': 'sgd', 'lr': 0.1, 'm': 0.9}))

    def best_from(fname, key, pick=None):
        try:
            data_j = json.load(open(os.path.join(od, fname)))[key]
            ok = [r for r in data_j if not r['diverged']]
            if pick:
                ok = [r for r in ok if r['params'].get('form') == pick]
            return ok[0] if ok else None
        except Exception as e:
            print(f"  (aviso: nao li {fname} [{key}]: {e})")
            return None

    b = best_from('tune_grid.json', 'adam_baseline')
    if b: configs.append(rebuild(b['params']))
    b = best_from('tune_confirm.json', 'confirmed')
    if b: configs.append(rebuild(b['params']))
    for form in ('sgd_classic', 'sgd_sep'):
        b = best_from('tune_sgdpid.json', 'confirmed', pick=form)
        if b: configs.append(rebuild(b['params']))

    res = run_final_with_energy(configs, data, args, seeds=args.seeds)
    save({'note': "val_mean/val_std = acuracia de TESTE; energia medida com "
                  "CodeCarbon (CPU+GPU), CO2 no fator BR 0.12 kg/kWh",
          'energia_medida': bool(args.energy and HAS_CC),
          'final_test': res}, os.path.join(od, 'final_results.json'))

    has_e = any('energy_wh_per_run' in r for r in res if not r['diverged'])
    print("\n==== TABELA FINAL (TESTE, media +/- desvio sobre seeds) ====")
    hdr = f"{'Otimizador':<32} {'Test acc (%)':>15} {'Ep->alvo':>9} {'Tempo(s)':>9}"
    if has_e:
        hdr += f" {'Wh/run':>8} {'CO2(g)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        if r['diverged']:
            print(f"{r['tag']:<32} {'diverged':>15}")
            continue
        acc = f"{r['val_mean']:.2f} +/- {r['val_std']:.2f}"
        ep = (f"{r['ep_mean']:.1f}" if r['ep_mean'] is not None else "n/a")
        line = f"{r['tag']:<32} {acc:>15} {ep:>9} {r['time_per_run_s']:>9.1f}"
        if has_e:
            line += f" {r.get('energy_wh_per_run', float('nan')):>8.3f} {r.get('co2_g_per_run', float('nan')):>8.4f}"
        print(line)
    return res


# --------------------------------- main ------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['grid', 'confirm', 'dterm', 'sgdpid', 'final', 'all'],
                   default='all')
    p.add_argument('--epochs', type=int, default=8, help='orcamento por config (ranking se preserva)')
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--hidden', type=int, default=1000)
    p.add_argument('--target', type=float, default=None,
                   help='alvo (frac) p/ epocas-ate-alvo; auto por dataset se omitido')
    p.add_argument('--clip', type=float, default=0.0)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--kp-grid', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument('--ki-grid', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    p.add_argument('--kd-grid', type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5])
    p.add_argument('--adam-lr', type=float, default=1e-3)
    p.add_argument('--adam-lr-grid', type=float, nargs='+', default=[1e-3, 2e-3, 3e-3, 5e-3])
    p.add_argument('--sgd-lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--dataset', choices=list(DATASETS), default='mnist',
                   help='mnist | fashion | emnist (todos 28x28, MLP identica)')
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--out-dir', default=None,
                   help='default: ./experiments/tune_<dataset>')
    p.add_argument('--energy', action='store_true', default=True,
                   help='medir energia/CO2 na fase final (CodeCarbon)')
    p.add_argument('--no-energy', dest='energy', action='store_false',
                   help='desliga a medicao de energia')
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f'./experiments/tune_{args.dataset}'
    os.makedirs(args.out_dir, exist_ok=True)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {args.device}  | dataset: {args.dataset} | out: {args.out_dir}")
    t0 = time.time()
    train, val, test, num_classes = load_data_with_val(
        args.dataset, args.data_dir, args.device)
    data = (train, val, test)
    args.num_classes = num_classes
    # per-dataset default target (MNIST saturates ~98%, Fashion ~88-90%, EMNIST ~80%)
    if args.target is None:
        args.target = {'mnist': 0.96, 'fashion': 0.87, 'emnist': 0.78}[args.dataset]
    print(f"train {tuple(data[0][0].shape)}  val {tuple(data[1][0].shape)}  "
          f"test {tuple(data[2][0].shape)}  | classes {num_classes} | alvo {args.target}")

    if args.phase in ('grid', 'all'):
        ranked, _ = phase_grid(data, args)
    if args.phase in ('confirm', 'all'):
        confirmed = phase_confirm(data, args, ranked if args.phase == 'all' else None)
    if args.phase in ('dterm', 'all'):
        phase_dterm(data, args)
    if args.phase == 'sgdpid':
        phase_sgdpid(data, args)
    if args.phase == 'final':
        phase_final(data, args)

    print(f"\nconcluido em {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()

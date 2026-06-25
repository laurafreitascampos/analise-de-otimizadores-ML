"""
Compare SGD, Adam, SGD_PID and Adam_PID on the MNIST + MLP task.

This is Dai et al. (2023) Experiment 4.1: a single-hidden-layer MLP
(784 -> 1000 -> 10, ReLU, softmax) on MNIST. It is tiny, so the whole
dataset is loaded ONCE into device memory and minibatches are taken by
indexing GPU tensors. There is no DataLoader and no worker process, which
removes the deadlock you hit on the ResNet run and makes each epoch a
fraction of a second. Multi-seed mean +/- std and the comparison plot are
kept.

Run:
    source .venv/bin/activate.fish
    python compare_mnist_mlp.py --epochs 30 --seeds 0 1 2

Faithful-to-paper knobs: batch size 100; Adam family lr defaults shown below.
Dai et al. used lr (eta) = 1e-4 and 100 epochs; pass --lr 1e-4 --epochs 100
to match exactly (slower to converge, but that is what the paper reports).

Requires: torch, torchvision, numpy, matplotlib.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from pid_optimizers import SGD_PID, Adam_PID

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# ----------------------------- reproducibility ----------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------- model ------------------------------------- #
def make_mlp(hidden=1000):
    """784 -> hidden (ReLU) -> 10. softmax is folded into CrossEntropyLoss."""
    return nn.Sequential(
        nn.Linear(28 * 28, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    )


# ------------------------------- data -------------------------------------- #
def load_mnist_to_device(data_dir, device):
    """Load MNIST fully into device memory as flat, normalized tensors."""
    import torchvision
    train = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    def prep(ds):
        x = ds.data.float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD)
        x = x.view(x.size(0), -1).to(device)          # (N, 784)
        y = ds.targets.to(device)                     # (N,)
        return x, y

    return prep(train), prep(test)


# ----------------------------- optimizer factory --------------------------- #
def make_optimizer(name, params, args):
    name = name.lower()
    wd = args.weight_decay
    if name == 'sgd':
        lr = args.lr if args.lr is not None else 0.1
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == 'adam':
        lr = args.lr if args.lr is not None else 1e-3
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == 'sgd_pid':
        lr = args.lr if args.lr is not None else 0.1
        return SGD_PID(params, lr=lr, momentum=0.9, weight_decay=wd,
                       Kp=args.kp, Ki=args.ki, Kd=args.kd)
    if name == 'adam_pid':
        lr = args.lr if args.lr is not None else 1e-3
        return Adam_PID(params, lr=lr, weight_decay=wd,
                        Kp=args.kp, Ki=args.ki, Kd=args.kd)
    raise ValueError(f"Unknown optimizer: {name}")


# ------------------------------- eval -------------------------------------- #
@torch.no_grad()
def evaluate(model, X, Y, criterion, batch=1000):
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        out = model(xb)
        loss_sum += criterion(out, yb).item() * yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / X.size(0), correct / X.size(0)


# --------------------- single (optimizer, seed) run ------------------------ #
def run_once(name, seed, args, data, device):
    (Xtr, Ytr), (Xte, Yte) = data
    set_seed(seed)
    model = make_mlp(args.hidden).to(device)
    opt = make_optimizer(name, model.parameters(), args)
    criterion = nn.CrossEntropyLoss()

    hist = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
            'epochs_to_target': None, 'diverged': False, 'seed': seed}
    N, bs = Xtr.size(0), args.batch_size

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N, device=device)
        run_loss, run_correct = 0.0, 0

        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"  !! non-finite loss ({loss_val}) at epoch {epoch+1}. "
                      f"Stopping '{name}' seed {seed}.")
                hist['diverged'] = True
                break

            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            run_loss += loss_val * yb.size(0)
            run_correct += (out.argmax(1) == yb).sum().item()

        if hist['diverged']:
            break

        tr_loss, tr_acc = run_loss / N, run_correct / N
        te_loss, te_acc = evaluate(model, Xte, Yte, criterion)
        hist['train_loss'].append(tr_loss)
        hist['train_acc'].append(tr_acc)
        hist['test_loss'].append(te_loss)
        hist['test_acc'].append(te_acc)
        if hist['epochs_to_target'] is None and te_acc >= args.target:
            hist['epochs_to_target'] = epoch + 1

        print(f"[{name:>8} s{seed}] epoch {epoch+1:3d}/{args.epochs} "
              f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test_loss {te_loss:.4f} acc {te_acc:.4f} | {time.time()-t0:.2f}s")

    return hist


# ----------------- aggregate a list of per-seed histories ------------------ #
def aggregate(seed_hists):
    full = [h for h in seed_hists if not h['diverged'] and h['test_acc']]
    agg = {'n_seeds': len(seed_hists), 'n_completed': len(full),
           'final_acc': [h['test_acc'][-1] for h in full],
           'best_acc': [max(h['test_acc']) for h in full],
           'epochs_to_target': [h['epochs_to_target'] for h in full],
           'curves': {}}
    if not full:
        return agg
    n_ep = min(len(h['test_acc']) for h in full)
    for key in ('train_loss', 'test_loss', 'train_acc', 'test_acc'):
        arr = np.array([h[key][:n_ep] for h in full])
        agg['curves'][key] = {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist()}
    return agg


def fmt_mean_std(values):
    if not values:
        return "diverged"
    a = np.array(values, dtype=float)
    return f"{a.mean():.4f} +/- {a.std():.4f}"


def fmt_ett(values):
    reached = [v for v in values if v is not None]
    if not values:
        return "-"
    if not reached:
        return f"0/{len(values)} reached"
    return f"{len(reached)}/{len(values)} @ {np.mean(reached):.1f} ep"


# ------------------------------- plotting ---------------------------------- #
def plot_results(results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [('train_loss', 'Train loss'), ('test_loss', 'Test loss'),
              ('train_acc', 'Train accuracy'), ('test_acc', 'Test accuracy')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name, agg in results.items():
            curve = agg['curves'].get(key)
            if not curve:
                continue
            mean = np.array(curve['mean'])
            std = np.array(curve['std'])
            x = np.arange(1, len(mean) + 1)
            line, = ax.plot(x, mean, label=name)
            if len(std) and std.any():
                ax.fill_between(x, mean - std, mean + std, alpha=0.2,
                                color=line.get_color())
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'comparison_mnist_mlp.png')
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


# --------------------------------- main ------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizers', nargs='+',
                   default=['sgd', 'adam', 'sgd_pid', 'adam_pid'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=100)     # Dai et al.
    p.add_argument('--hidden', type=int, default=1000)        # Dai et al.
    p.add_argument('--lr', type=float, default=None,
                   help='override lr for ALL optimizers; Dai used 1e-4')
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--kp', type=float, default=0.5)
    p.add_argument('--ki', type=float, default=1.0)
    p.add_argument('--kd', type=float, default=0.3)
    p.add_argument('--clip', type=float, default=0.0,
                   help='grad-norm clip; 0 disables (MLP is stable without it)')
    p.add_argument('--target', type=float, default=0.98)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--out-dir', default='./experiments/mnist_mlp_compare')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"Seeds: {args.seeds} | MLP 784-{args.hidden}-10 | batch {args.batch_size}")

    data = load_mnist_to_device(args.data_dir, device)
    print(f"Loaded MNIST: train {tuple(data[0][0].shape)}, test {tuple(data[1][0].shape)}")

    results, raw = {}, {}
    for name in args.optimizers:
        print(f"\n=== {name} ===")
        seed_hists = [run_once(name, s, args, data, device) for s in args.seeds]
        raw[name] = seed_hists
        results[name] = aggregate(seed_hists)

    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(args), 'aggregate': results, 'per_seed': raw}, f, indent=2)

    print("\n============ summary (mean +/- std over seeds) ============")
    print(f"{'optimizer':>10} | {'best test acc':>22} | {'final test acc':>22} | epochs->{args.target}")
    for name, agg in results.items():
        print(f"{name:>10} | {fmt_mean_std(agg['best_acc']):>22} | "
              f"{fmt_mean_std(agg['final_acc']):>22} | {fmt_ett(agg['epochs_to_target'])}")

    plot_results(results, args.out_dir)


if __name__ == '__main__':
    main()

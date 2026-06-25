"""
Compare SGD, Adam, SGD_PID and Adam_PID training a ResNet-18 on CIFAR-10.

Reproduces the Dai et al. (2023) ResNet-18 / CIFAR-10 setting and adds the
other three optimizers under an identical pipeline, with optional multi-seed
runs reporting mean +/- std.

Robustness (after the dataloader deadlock at an epoch boundary):
  * conservative dataloader (no persistent_workers, prefetch_factor=2,
    default --workers 4) -> avoids the multiprocessing deadlock
  * per-epoch checkpoint + automatic resume -> a crash costs <= 1 epoch;
    just rerun the SAME command and it continues from the last checkpoint
  * watchdog -> if no batch makes progress for --watchdog seconds the process
    exits with a clear message instead of hanging forever (resume on rerun)

GTX 16-series: AMP (fp16) is OFF by default. --fast enables cudnn.benchmark.

Smoke test:               python compare_resnet18.py --fast --smoke
Multi-seed full run:      python compare_resnet18.py --fast --epochs 40 --seeds 0 1 2
Start over (new gains):   add --fresh   (or use a new --out-dir)

Tip: run inside tmux so a dropped SSH/terminal does not kill it:
    tmux new -s pid
    python compare_resnet18.py --fast --epochs 40 --seeds 0 1 2

Requires: torch, torchvision, numpy, matplotlib.
"""

import argparse
import json
import math
import os
import random
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from pid_optimizers import SGD_PID, Adam_PID


# ------------------------------- watchdog ---------------------------------- #
class Watchdog:
    """Hard-exits the process if no heartbeat arrives within `timeout` seconds.
    The hang we hit lives inside the dataloader's C/multiprocessing layer and
    cannot be interrupted cleanly from Python, so we exit and rely on the
    checkpoint to resume."""
    def __init__(self, timeout):
        self.timeout = timeout
        self.last = time.time()
        self.enabled = timeout and timeout > 0
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def beat(self):
        self.last = time.time()

    def _loop(self):
        while True:
            time.sleep(5)
            if time.time() - self.last > self.timeout:
                print(f"\n!! Watchdog: no progress for {self.timeout}s "
                      f"(likely a dataloader deadlock). Exiting. Rerun the SAME "
                      f"command to resume from the last checkpoint.", flush=True)
                os._exit(2)


# ----------------------------- reproducibility ----------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state():
    return {'py': random.getstate(), 'np': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    try:
        random.setstate(s['py'])
        np.random.set_state(s['np'])
        torch.set_rng_state(s['torch'])
        if s.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s['cuda'])
    except Exception as e:
        print(f"  (warning: could not fully restore RNG state: {e})")


# ------------------------------- model ------------------------------------- #
def make_resnet18(stem='cifar', num_classes=10):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if stem == 'cifar':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


# ------------------------------- data -------------------------------------- #
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def make_loaders(data_dir, batch_size, workers, use_cuda):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    # Conservative settings: NO persistent_workers (it was deadlocking at the
    # epoch/seed boundary), modest prefetch. 4 workers already feed this GPU
    # with room to spare (you saw data ~0.3s vs compute ~87s).
    common = dict(num_workers=workers, pin_memory=use_cuda)
    if workers > 0:
        common.update(prefetch_factor=2)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=False, **common)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **common)
    return train_loader, test_loader


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
def evaluate(model, loader, criterion, device, wd):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        wd.beat()
    return loss_sum / total, correct / total


# --------------------------- checkpoint paths ------------------------------ #
def ckpt_path(out_dir, name, seed):
    return os.path.join(out_dir, f"ckpt_{name}_s{seed}.pt")


def done_path(out_dir, name, seed):
    return os.path.join(out_dir, f"done_{name}_s{seed}.json")


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------- single (optimizer, seed) run ------------------------ #
def run_once(name, seed, args, device, wd):
    dp = done_path(args.out_dir, name, seed)
    if not args.smoke and os.path.exists(dp):
        with open(dp) as f:
            print(f"[{name} s{seed}] already complete -> loaded from disk")
            return json.load(f)

    set_seed(seed)
    train_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.workers, device.type == 'cuda')

    model = make_resnet18(stem=args.stem).to(device)
    opt = make_optimizer(name, model.parameters(), args)
    criterion = nn.CrossEntropyLoss()
    amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=amp)

    hist = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
            'epochs_to_target': None, 'diverged': False, 'seed': seed}
    start_epoch = 0

    cp = ckpt_path(args.out_dir, name, seed)
    if not args.smoke and not args.fresh and os.path.exists(cp):
        ck = torch.load(cp, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        set_rng_state(ck['rng'])
        hist = ck['hist']
        start_epoch = ck['epoch'] + 1
        print(f"[{name} s{seed}] resuming from epoch {start_epoch}")

    limit = 30 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, data_t, compute_t = time.time(), 0.0, 0.0
        run_loss, run_correct, run_total = 0.0, 0, 0
        wd.beat()
        t_batch = time.time()

        for i, (x, y) in enumerate(train_loader):
            if limit is not None and i >= limit:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            data_t += time.time() - t_batch
            t_compute = time.time()

            opt.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=amp):
                out = model(x)
                loss = criterion(out, y)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"  !! non-finite loss ({loss_val}) at epoch {epoch+1} batch {i}. "
                      f"Stopping '{name}' seed {seed}.")
                hist['diverged'] = True
                break

            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()

            run_loss += loss_val * y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            run_total += y.size(0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            compute_t += time.time() - t_compute
            wd.beat()
            t_batch = time.time()

        if hist['diverged']:
            break

        tr_loss, tr_acc = run_loss / run_total, run_correct / run_total
        te_loss, te_acc = evaluate(model, test_loader, criterion, device, wd)
        hist['train_loss'].append(tr_loss)
        hist['train_acc'].append(tr_acc)
        hist['test_loss'].append(te_loss)
        hist['test_acc'].append(te_acc)
        if hist['epochs_to_target'] is None and te_acc >= args.target:
            hist['epochs_to_target'] = epoch + 1

        print(f"[{name:>8} s{seed}] epoch {epoch+1:3d}/{epochs} "
              f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test_loss {te_loss:.4f} acc {te_acc:.4f} "
              f"| {time.time()-t0:.1f}s (data {data_t:.1f}s / compute {compute_t:.1f}s)")

        if not args.smoke:
            atomic_save({'model': model.state_dict(), 'opt': opt.state_dict(),
                         'scaler': scaler.state_dict(), 'rng': get_rng_state(),
                         'epoch': epoch, 'hist': hist}, cp)
        wd.beat()

    # finished (or diverged): record and drop the resume checkpoint
    if not args.smoke:
        with open(dp, 'w') as f:
            json.dump(hist, f)
        if os.path.exists(cp):
            os.remove(cp)
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
    path = os.path.join(out_dir, 'comparison.png')
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


# --------------------------------- main ------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizers', nargs='+',
                   default=['sgd', 'adam', 'sgd_pid', 'adam_pid'])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=None,
                   help='override learning rate for ALL optimizers')
    p.add_argument('--weight-decay', type=float, default=5e-4,
                   help='Dai et al. used none; pass 0 for a faithful Adam_PID run')
    p.add_argument('--kp', type=float, default=0.5)
    p.add_argument('--ki', type=float, default=1.0)
    p.add_argument('--kd', type=float, default=0.3)
    p.add_argument('--clip', type=float, default=1.0, help='grad-norm clip; 0 disables')
    p.add_argument('--target', type=float, default=0.85)
    p.add_argument('--stem', choices=['cifar', 'imagenet'], default='cifar')
    p.add_argument('--seeds', type=int, nargs='+', default=[0],
                   help='one or more seeds, e.g. --seeds 0 1 2')
    p.add_argument('--workers', type=int, default=4,
                   help='conservative default; raise only if data time dominates')
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--out-dir', default='./experiments/resnet18_compare')
    p.add_argument('--fast', action='store_true', help='enable cudnn.benchmark')
    p.add_argument('--amp', action='store_true',
                   help='fp16 mixed precision (NOT recommended on GTX 16xx)')
    p.add_argument('--watchdog', type=int, default=180,
                   help='seconds without batch progress before aborting; 0 disables')
    p.add_argument('--fresh', action='store_true',
                   help='ignore existing checkpoints/done files and start over')
    p.add_argument('--smoke', action='store_true', help='quick 1-epoch pipeline check')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.fast and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"Seeds: {args.seeds} | workers: {args.workers} | watchdog: {args.watchdog}s")
    if args.amp and device.type == 'cuda':
        print("AMP is ON: usual source of nan on a GTX 16-series; drop --amp if you see it.")

    if args.fresh:
        for name in args.optimizers:
            for s in args.seeds:
                for path in (ckpt_path(args.out_dir, name, s), done_path(args.out_dir, name, s)):
                    if os.path.exists(path):
                        os.remove(path)
        print("Cleared previous checkpoints/done files (--fresh).")

    wd = Watchdog(args.watchdog)

    results, raw = {}, {}
    for name in args.optimizers:
        print(f"\n=== {name} ===")
        seed_hists = [run_once(name, s, args, device, wd) for s in args.seeds]
        raw[name] = seed_hists
        results[name] = aggregate(seed_hists)

    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(args), 'aggregate': results, 'per_seed': raw}, f, indent=2)

    print("\n===================== summary (mean +/- std over seeds) =====================")
    print(f"{'optimizer':>10} | {'best test acc':>22} | {'final test acc':>22} | epochs->{args.target}")
    for name, agg in results.items():
        print(f"{name:>10} | {fmt_mean_std(agg['best_acc']):>22} | "
              f"{fmt_mean_std(agg['final_acc']):>22} | {fmt_ett(agg['epochs_to_target'])}")

    if not args.smoke:
        plot_results(results, args.out_dir)


if __name__ == '__main__':
    main()
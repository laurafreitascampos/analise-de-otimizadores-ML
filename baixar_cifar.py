"""
get_cifar10.py
Baixa o CIFAR-10 (versao Python), processa e salva em cifar10.npz.

Diferente do MNIST (um unico pickle .gz), o CIFAR-10 vem como um .tar.gz
contendo seis arquivos pickled: data_batch_1..5 (treino) e test_batch (teste).
Cada um guarda as imagens como uma matriz (N, 3072) uint8, no layout
[1024 R | 1024 G | 1024 B] por imagem -> reshape para (N, 3, 32, 32).

Formato de saida (cifar10.npz):
  x_train (45000, 3, 32, 32) uint8     y_train (45000,) int64
  x_val   ( 5000, 3, 32, 32) uint8     y_val   ( 5000,) int64   # se VAL_SIZE>0
  x_test  (10000, 3, 32, 32) uint8     y_test  (10000,) int64

Imagens ficam em uint8 [0,255] (arquivo menor e sem perda). Normalize no loader
usando as constantes oficiais por canal abaixo.
"""
import os
import pickle
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR = "cifar-10-python.tar.gz"
OUT = "cifar10.npz"
VAL_SIZE = 5000     # imagens de treino reservadas p/ validacao (0 = sem split de val)
SEED = 0

# Media / desvio por canal (R, G, B) do CIFAR-10 -- use no loader p/ normalizar.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def download(url=URL, dst=TAR):
    if os.path.exists(dst):
        print(f"{dst} ja existe, pulando download.")
        return dst
    print("Baixando CIFAR-10 (~170 MB)...")
    urllib.request.urlretrieve(url, dst)
    print("OK ->", dst)
    return dst


def _load_batch(fileobj):
    """Le um pickle de batch e devolve (x uint8 (N,3,32,32), y int64 (N,))."""
    d = pickle.load(fileobj, encoding="latin1")  # chaves viram strings normais
    x = d["data"].astype(np.uint8).reshape(-1, 3, 32, 32)
    y = np.array(d["labels"], dtype=np.int64)
    return x, y


def load_from_tar(tar_path=TAR):
    """Extrai os batches do .tar.gz e devolve (x_tr, y_tr, x_te, y_te)."""
    train_batches = {}
    x_te = y_te = None
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if name.startswith("data_batch_"):
                train_batches[name] = _load_batch(tar.extractfile(m))
            elif name == "test_batch":
                x_te, y_te = _load_batch(tar.extractfile(m))

    if not train_batches or x_te is None:
        raise RuntimeError("Batches do CIFAR-10 nao encontrados no tar.")

    # concatena data_batch_1 ... data_batch_5 em ordem deterministica
    xs, ys = [], []
    for name in sorted(train_batches):
        x, y = train_batches[name]
        xs.append(x)
        ys.append(y)
    x_tr = np.concatenate(xs)
    y_tr = np.concatenate(ys)
    return x_tr, y_tr, x_te, y_te


def main():
    download()
    x_tr, y_tr, x_te, y_te = load_from_tar()

    if VAL_SIZE > 0:
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(x_tr))
        val_idx, tr_idx = idx[:VAL_SIZE], idx[VAL_SIZE:]
        x_val, y_val = x_tr[val_idx], y_tr[val_idx]
        x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
    else:
        x_val = np.empty((0, 3, 32, 32), np.uint8)
        y_val = np.empty((0,), np.int64)

    np.savez_compressed(
        OUT,
        x_train=x_tr, y_train=y_tr,
        x_val=x_val,  y_val=y_val,
        x_test=x_te,  y_test=y_te,
    )
    print(f"OK -> {OUT}")
    print(f"  treino {x_tr.shape}  val {x_val.shape}  teste {x_te.shape}")


if __name__ == "__main__":
    main()
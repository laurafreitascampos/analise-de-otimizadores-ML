"""Baixa a base MNIST e salva em mnist.npz (50k treino / 10k val / 10k teste)."""
import io, gzip, pickle, urllib.request
import numpy as np

URL = ("https://github.com/mnielsen/neural-networks-and-deep-learning/"
       "raw/master/data/mnist.pkl.gz")

print("Baixando MNIST...")
raw = urllib.request.urlopen(URL, timeout=60).read()
with gzip.open(io.BytesIO(raw), "rb") as f:
    tr, va, te = pickle.load(f, encoding="latin1")

np.savez_compressed(
    "mnist.npz",
    x_train=tr[0].astype(np.float32), y_train=tr[1].astype(np.int64),
    x_val=va[0].astype(np.float32),   y_val=va[1].astype(np.int64),
    x_test=te[0].astype(np.float32),  y_test=te[1].astype(np.int64),
)
print("OK -> mnist.npz", tr[0].shape, te[0].shape)

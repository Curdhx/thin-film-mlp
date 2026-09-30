"""Numpy-only multilayer perceptron with Adam optimizer.

Architecture: 4 -> 128 -> 128 -> 64 -> 41 (ReLU hidden, sigmoid output)
Input  : 4 normalized thicknesses (40-180 nm mapped to [0, 1])
Output : 41-point reflectance spectrum (values in [0, 1])
Loss   : mean squared error (MSE)
"""
import numpy as np


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class MLP:
    def __init__(self, sizes, seed):
        self.sizes = list(sizes)
        rng = np.random.RandomState(seed)
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            # He initialization (suited for ReLU)
            self.W.append(rng.randn(sizes[i], sizes[i + 1]) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros(sizes[i + 1]))
        self._init_adam()

    def _init_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        """X (N, 4) -> sigmoid output (N, 41), plus pre-activations Zs."""
        A = X
        Zs = []
        L = len(self.W)
        for i in range(L - 1):
            Z = A @ self.W[i] + self.b[i]
            Zs.append(Z)
            A = relu(Z)
        Z = A @ self.W[-1] + self.b[-1]
        Zs.append(Z)
        return sigmoid(Z), Zs

    def backward(self, X, T):
        """MSE gradient. Returns (grads_W, grads_b)."""
        Y, Zs = self.forward(X)
        N = X.shape[0]
        L = len(self.W)
        gW = [None] * L
        gb = [None] * L
        # output layer (sigmoid + MSE)
        dZ = (Y - T) * Y * (1.0 - Y) / N
        A_prev = relu(Zs[L - 2])
        gW[L - 1] = A_prev.T @ dZ
        gb[L - 1] = dZ.sum(axis=0)
        # hidden layers (ReLU)
        for i in range(L - 2, -1, -1):
            dA = dZ @ self.W[i + 1].T
            dZ = dA * (Zs[i] > 0)
            A_prev = X if i == 0 else relu(Zs[i - 1])
            gW[i] = A_prev.T @ dZ
            gb[i] = dZ.sum(axis=0)
        return gW, gb

    def step(self, gW, gb, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """One Adam update."""
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (gW[i] ** 2)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb[i] ** 2)
            mhW = self.mW[i] / (1 - beta1 ** self.t)
            vhW = self.vW[i] / (1 - beta2 ** self.t)
            mhb = self.mb[i] / (1 - beta1 ** self.t)
            vhb = self.vb[i] / (1 - beta2 ** self.t)
            self.W[i] -= lr * mhW / (np.sqrt(vhW) + eps)
            self.b[i] -= lr * mhb / (np.sqrt(vhb) + eps)

    def train(self, X, T, Xval=None, Tval=None, epochs=500, batch_size=128,
              lr=1e-3, seed=0, verbose=True):
        """Mini-batch Adam training. Returns dict {'train': [...], 'val': [...]}."""
        rng = np.random.RandomState(seed)
        N = X.shape[0]
        hist = {"train": [], "val": []}
        for ep in range(epochs):
            idx = rng.permutation(N)
            for b in range(0, N, batch_size):
                bi = idx[b:b + batch_size]
                gW, gb = self.backward(X[bi], T[bi])
                self.step(gW, gb, lr)
            tr = float(np.mean((self.forward(X)[0] - T) ** 2))
            hist["train"].append(tr)
            if Xval is not None:
                va = float(np.mean((self.forward(Xval)[0] - Tval) ** 2))
                hist["val"].append(va)
            if verbose and (ep + 1) % 100 == 0:
                msg = "  epoch %4d: train MSE %.6f" % (ep + 1, tr)
                if Xval is not None:
                    msg += ", val MSE %.6f" % va
                print(msg)
        return hist

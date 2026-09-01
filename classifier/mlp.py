"""Small regularised MLP for the classifier slot — PyTorch, CPU, <1 MB.

sklearn-ish surface (`fit`, `predict_proba`, `classes_`) so it drops into the
same {clf, classes, clip, features} bundle and the eval harness scores it like
any other model.

Regularisation for the small honest test: BatchNorm + Dropout + weight decay +
label smoothing + early stopping on the validation split. Post-fit
`set_temperature(Xval, yval)` learns a single scaling scalar (Guo et al.,
"On Calibration of Modern Neural Networks", ICML 2017) — divides the logits,
does not change the argmax, fixes the over-confidence.
"""

from __future__ import annotations

import numpy as np


class TorchMLP:
    def __init__(self, n_classes: int, hidden=(256, 128), dropout=0.3,
                 lr=1e-3, weight_decay=1e-4, label_smoothing=0.05,
                 max_epochs=200, patience=15, batch_size=512,
                 class_weight: dict | None = None, seed=0):
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.seed = seed
        self.classes_ = np.arange(n_classes)
        self.temperature = 1.0
        self._net = None
        self._mu = self._sd = None

    # ---- torch net ----
    def _build(self, in_dim):
        import torch.nn as nn

        layers, d = [], in_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(self.dropout)]
            d = h
        layers += [nn.Linear(d, self.n_classes)]
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.int64)
        self._mu = X.mean(0)
        self._sd = X.std(0) + 1e-6
        Xs = (X - self._mu) / self._sd

        if X_val is None:                       # carve one if not given
            n = len(X); k = max(1, int(0.1 * n))
            perm = np.random.permutation(n)
            X_val, y_val = X[perm[:k]], y[perm[:k]]
            Xs, y = Xs[perm[k:]], y[perm[k:]]
        Xv = ((np.asarray(X_val, np.float32) - self._mu) / self._sd)
        yv = np.asarray(y_val, np.int64)

        self._net = self._build(Xs.shape[1])
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        w = None
        if self.class_weight:
            w = torch.tensor([self.class_weight.get(i, 1.0)
                              for i in range(self.n_classes)], dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=self.label_smoothing)

        Xt = torch.tensor(Xs); yt = torch.tensor(y)
        Xvt = torch.tensor(Xv); yvt = torch.tensor(yv)
        best, best_state, bad = 1e9, None, 0
        for _ in range(self.max_epochs):
            self._net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(self._net(Xt[b]), yt[b]).backward()
                opt.step()
            self._net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self._net(Xvt), yvt))
            if vloss < best - 1e-4:
                best, best_state, bad = vloss, {k: v.clone() for k, v in
                                               self._net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self._net.load_state_dict(best_state)
        self.set_temperature(X_val, y_val)
        return self

    def _logits(self, X):
        import torch

        Xs = (np.asarray(X, np.float32) - self._mu) / self._sd
        self._net.eval()
        with torch.no_grad():
            return self._net(torch.tensor(Xs)).numpy()

    def set_temperature(self, X_val, y_val):
        """1-D search for the T that minimises val NLL."""
        logits = self._logits(X_val)
        y = np.asarray(y_val, np.int64)
        best_t, best_nll = 1.0, 1e9
        for t in np.linspace(0.5, 5.0, 46):
            p = _softmax(logits / t)
            nll = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        self.temperature = best_t
        return best_t

    def predict_proba(self, X):
        return _softmax(self._logits(X) / self.temperature)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def demo():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (2000, 20)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)  # 3 classes
    m = TorchMLP(n_classes=3, hidden=(64, 32), max_epochs=80, patience=12,
                 seed=0).fit(X[:1600], y[:1600], X[1600:], y[1600:])
    p = m.predict_proba(X[1600:])
    assert p.shape == (400, 3) and np.allclose(p.sum(1), 1, atol=1e-4)
    acc = (p.argmax(1) == y[1600:]).mean()
    base = np.bincount(y[1600:]).max() / 400          # majority-class baseline
    assert acc > base + 0.1, (acc, base)             # beats majority by a margin
    assert 0.5 <= m.temperature <= 5.0
    print(f"mlp.demo OK  acc {acc:.2f}  T {m.temperature:.2f}")


if __name__ == "__main__":
    demo()

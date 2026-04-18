"""
2D Poisson equation on the unit square:

    -(u_xx + u_yy) = f(x, y)     (x, y) in (0, 1)^2
    u = 0                        on the boundary

with source term  f(x, y) = 2 * pi^2 * sin(pi x) * sin(pi y),
the analytical solution is  u(x, y) = sin(pi x) * sin(pi y).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pinn import PINN
from grad import d2


def source(x, y):
    return 2 * (np.pi ** 2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)


def u_true(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def sample_boundary(n):
    # uniform points on each of the 4 sides
    a = torch.rand(n, 1)
    zero = torch.zeros(n, 1)
    one = torch.ones(n, 1)
    # bottom, top, left, right
    xb = torch.cat([a, a, zero, one], dim=0)
    yb = torch.cat([zero, one, a, a], dim=0)
    return xb, yb


def train(epochs=8000, lr=1e-3, w_bc=20.0):
    model = PINN(in_dim=2, hidden=60, depth=5)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_f = torch.rand(8000, 1, requires_grad=True)
    y_f = torch.rand(8000, 1, requires_grad=True)
    xb, yb = sample_boundary(300)

    for ep in range(epochs):
        opt.zero_grad()

        u = model(x_f, y_f)
        u_xx = d2(u, x_f)
        u_yy = d2(u, y_f)
        pde = ((-(u_xx + u_yy) - source(x_f, y_f)) ** 2).mean()

        bc = (model(xb, yb) ** 2).mean()

        loss = pde + w_bc * bc
        loss.backward()
        opt.step()

        if ep % 800 == 0:
            print(f"epoch {ep:5d}  loss={loss.item():.6f}  "
                  f"pde={pde.item():.3e}  bc={bc.item():.3e}")

    return model


@torch.no_grad()
def plot_solution(model, save="figures/poisson.png"):
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
    u_pred = model(xy[:, 0:1], xy[:, 1:2]).numpy().reshape(X.shape)
    u_ex = u_true(X, Y)
    err = np.abs(u_pred - u_ex)
    print(f"L2 error: {np.sqrt((err**2).mean()):.5f}  max error: {err.max():.5f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, data, title in zip(ax, [u_ex, u_pred, err], ["analytical", "PINN", "|error|"]):
        im = a.imshow(data, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
        a.set_title(title); a.set_xlabel("x"); a.set_ylabel("y")
        plt.colorbar(im, ax=a)
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    print(f"saved {save}")


@torch.no_grad()
def export_json(model, path):
    import json
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
    u_pred = model(xy[:, 0:1], xy[:, 1:2]).numpy().reshape(X.shape)
    u_ex = u_true(X, Y)
    err = np.abs(u_pred - u_ex)
    data = {
        "name": "poisson",
        "x": x.tolist(),
        "y": y.tolist(),
        "u_pinn": u_pred.tolist(),
        "u_exact": u_ex.tolist(),
        "l2_error": float(np.sqrt((err**2).mean())),
        "max_error": float(err.max()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"exported {path}")


if __name__ == "__main__":
    torch.manual_seed(2)
    np.random.seed(2)
    os.makedirs("figures", exist_ok=True)
    model = train()
    plot_solution(model)
    export_json(model, "data/poisson.json")

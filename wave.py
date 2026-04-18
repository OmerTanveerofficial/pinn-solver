"""
1D wave equation:   u_tt = c^2 * u_xx
Domain: x in [0, 1], t in [0, 1]
BC:     u(0, t) = u(1, t) = 0
IC:     u(x, 0) = sin(pi * x),  u_t(x, 0) = 0

Analytical solution:  u(x, t) = cos(c * pi * t) * sin(pi * x)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pinn import PINN
from grad import d, d2


C = 1.0


def u_true(x, t):
    return np.cos(C * np.pi * t) * np.sin(np.pi * x)


def train(epochs=6000, lr=1e-3, w_bc=10.0, w_ic=20.0):
    model = PINN(in_dim=2, hidden=60, depth=5)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # collocation / boundary / initial points
    x_f = torch.rand(10000, 1, requires_grad=True)
    t_f = torch.rand(10000, 1, requires_grad=True)

    t_b = torch.rand(300, 1)
    x_b0 = torch.zeros(300, 1)
    x_b1 = torch.ones(300, 1)

    x_i = torch.rand(300, 1, requires_grad=True)
    t_i = torch.zeros(300, 1, requires_grad=True)
    u_i_target = torch.sin(np.pi * x_i.detach())

    for ep in range(epochs):
        opt.zero_grad()

        # pde: u_tt - c^2 u_xx = 0
        u = model(x_f, t_f)
        u_tt = d2(u, t_f)
        u_xx = d2(u, x_f)
        pde = ((u_tt - C**2 * u_xx) ** 2).mean()

        # boundary
        bc = (model(x_b0, t_b) ** 2).mean() + (model(x_b1, t_b) ** 2).mean()

        # initial: u(x, 0) = sin(pi x)
        u_i = model(x_i, t_i)
        ic_u = ((u_i - u_i_target) ** 2).mean()

        # initial: u_t(x, 0) = 0 -- need derivative wrt t at t=0
        u_t_i = d(u_i, t_i)
        ic_ut = (u_t_i ** 2).mean()

        loss = pde + w_bc * bc + w_ic * (ic_u + ic_ut)
        loss.backward()
        opt.step()

        if ep % 500 == 0:
            print(f"epoch {ep:5d}  loss={loss.item():.6f}  "
                  f"pde={pde.item():.3e}  bc={bc.item():.3e}  "
                  f"ic_u={ic_u.item():.3e}  ic_ut={ic_ut.item():.3e}")

    return model


@torch.no_grad()
def plot_solution(model, save="figures/wave.png"):
    nx, nt = 100, 100
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], -1), dtype=torch.float32)
    u_pred = model(xt[:, 0:1], xt[:, 1:2]).numpy().reshape(X.shape)
    u_ex = u_true(X, T)
    err = np.abs(u_pred - u_ex)
    print(f"L2 error: {np.sqrt((err**2).mean()):.5f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, data, title in zip(ax, [u_ex, u_pred, err], ["analytical", "PINN", "|error|"]):
        im = a.imshow(data, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="coolwarm")
        a.set_title(title); a.set_xlabel("x"); a.set_ylabel("t")
        plt.colorbar(im, ax=a)
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    print(f"saved {save}")


@torch.no_grad()
def export_json(model, path):
    import json
    nx, nt = 100, 100
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], -1), dtype=torch.float32)
    u_pred = model(xt[:, 0:1], xt[:, 1:2]).numpy().reshape(X.shape)
    u_ex = u_true(X, T)
    err = np.abs(u_pred - u_ex)
    data = {
        "name": "wave",
        "x": x.tolist(),
        "t": t.tolist(),
        "u_pinn": u_pred.tolist(),
        "u_exact": u_ex.tolist(),
        "l2_error": float(np.sqrt((err**2).mean())),
        "max_error": float(err.max()),
        "params": {"c": C},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"exported {path}")


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    os.makedirs("figures", exist_ok=True)
    model = train()
    plot_solution(model)
    export_json(model, "data/wave.json")

"""
1D heat equation:   u_t = alpha * u_xx
Domain: x in [0, 1], t in [0, 1]
BC:     u(0, t) = u(1, t) = 0
IC:     u(x, 0) = sin(pi * x)

Analytical solution:  u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from pinn import PINN
from grad import d, d2


ALPHA = 0.1


def u_true(x, t):
    return np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)


def sample_points(N_f=10000, N_b=200, N_i=200):
    # collocation points inside the domain, for the PDE residual
    x_f = torch.rand(N_f, 1, requires_grad=True)
    t_f = torch.rand(N_f, 1, requires_grad=True)

    # boundary points: x=0 and x=1, t random
    t_b = torch.rand(N_b, 1)
    x_b0 = torch.zeros(N_b, 1)
    x_b1 = torch.ones(N_b, 1)

    # initial points: t=0, x random
    x_i = torch.rand(N_i, 1)
    t_i = torch.zeros(N_i, 1)
    u_i = torch.sin(np.pi * x_i)

    return x_f, t_f, t_b, x_b0, x_b1, x_i, t_i, u_i


def train(epochs=5000, lr=1e-3):
    model = PINN(in_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_f, t_f, t_b, x_b0, x_b1, x_i, t_i, u_i = sample_points()

    losses = []
    for ep in range(epochs):
        opt.zero_grad()

        # pde residual
        u = model(x_f, t_f)
        u_t = d(u, t_f)
        u_xx = d2(u, x_f)
        pde = ((u_t - ALPHA * u_xx) ** 2).mean()

        # boundary loss
        u_b0 = model(x_b0, t_b)
        u_b1 = model(x_b1, t_b)
        bc = (u_b0 ** 2).mean() + (u_b1 ** 2).mean()

        # initial condition loss
        u_pred_i = model(x_i, t_i)
        ic = ((u_pred_i - u_i) ** 2).mean()

        loss = pde + bc + ic
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if ep % 500 == 0:
            print(f"epoch {ep:5d}  loss={loss.item():.6f}  "
                  f"pde={pde.item():.4e}  bc={bc.item():.4e}  ic={ic.item():.4e}")

    return model, losses


@torch.no_grad()
def evaluate(model, nx=100, nt=100):
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)

    xt = np.stack([X.ravel(), T.ravel()], axis=-1)
    xt_tensor = torch.tensor(xt, dtype=torch.float32)

    u_pred = model(xt_tensor[:, 0:1], xt_tensor[:, 1:2]).numpy().reshape(X.shape)
    u_exact = u_true(X, T)

    return X, T, u_pred, u_exact


def plot(X, T, u_pred, u_exact, save="figures/heat.png"):
    err = np.abs(u_pred - u_exact)
    l2 = np.sqrt((err ** 2).mean())
    print(f"L2 error: {l2:.5f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, data, title in zip(
        ax, [u_exact, u_pred, err],
        ["analytical", "PINN", "|error|"],
    ):
        im = a.imshow(data, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="viridis")
        a.set_title(title)
        a.set_xlabel("x")
        a.set_ylabel("t")
        plt.colorbar(im, ax=a)
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    print(f"saved {save}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    model, losses = train()
    X, T, u_pred, u_exact = evaluate(model)

    import os
    os.makedirs("figures", exist_ok=True)
    plot(X, T, u_pred, u_exact)

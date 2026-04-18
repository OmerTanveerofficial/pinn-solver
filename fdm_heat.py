"""
Classical finite-difference (Crank-Nicolson) solver for the 1D heat equation.
Used as a baseline to compare the PINN against.

Same problem as heat.py:
    u_t = alpha u_xx,  x in [0, 1],  t in [0, 1]
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(pi x)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


ALPHA = 0.1


def u_true(x, t):
    return np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)


def crank_nicolson(nx=101, nt=1001, alpha=ALPHA):
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = alpha * dt / (2 * dx ** 2)

    # initial condition
    u = np.sin(np.pi * x)
    history = [u.copy()]

    # Crank-Nicolson on interior points only (BC enforces u[0] = u[-1] = 0).
    # A u^{n+1}_int = B u^n_int + boundary contributions (which are 0 here).
    n_int = nx - 2
    # banded form for solve_banded: A has one subdiag and one superdiag
    ab = np.zeros((3, n_int))
    ab[0, 1:] = -r               # super-diagonal
    ab[1, :] = 1 + 2 * r         # main diagonal
    ab[2, :-1] = -r              # sub-diagonal

    # B as full matrix (small, easy to apply as a loop or with diag multiplications)
    for n in range(1, nt):
        u_int = u[1:-1]
        rhs = (1 - 2 * r) * u_int
        rhs[:-1] += r * u_int[1:]
        rhs[1:]  += r * u_int[:-1]
        u_new_int = solve_banded((1, 1), ab, rhs)

        u = np.zeros_like(u)
        u[1:-1] = u_new_int
        history.append(u.copy())

    return x, t, np.array(history)


def plot_fdm(x, t, U, save="figures/fdm_heat.png"):
    X, T = np.meshgrid(x, t)
    U_ex = u_true(X, T)
    err = np.abs(U - U_ex)
    print(f"FDM Crank-Nicolson L2 error: {np.sqrt((err**2).mean()):.5e}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, data, title in zip(ax, [U_ex, U, err], ["analytical", "FDM (CN)", "|error|"]):
        im = a.imshow(data, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="viridis")
        a.set_title(title); a.set_xlabel("x"); a.set_ylabel("t")
        plt.colorbar(im, ax=a)
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    print(f"saved {save}")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    x, t, U = crank_nicolson()
    plot_fdm(x, t, U)

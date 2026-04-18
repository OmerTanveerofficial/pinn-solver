# pinn-solver

Physics-Informed Neural Networks (PINNs) for solving PDEs.

A PINN is a small MLP that approximates the solution $u(x,t)$ of a PDE. Instead of training on data, you train it so its own derivatives (via autograd) satisfy the PDE and the boundary/initial conditions.

For the 1D heat equation $u_t = \alpha u_{xx}$, the loss looks like:

$$\mathcal{L} = \frac{1}{N_f} \sum |u_t - \alpha u_{xx}|^2 + \frac{1}{N_b} \sum |u(x_b, t_b)|^2 + \frac{1}{N_i} \sum |u(x_i, 0) - u_0(x_i)|^2$$

PDE residual + boundary loss + initial-condition loss.

This repo follows Raissi, Perdikaris, Karniadakis (2019): *Physics-informed neural networks*.

## Install

```
pip install -r requirements.txt
```

## Coming soon

- 1D heat equation
- 1D wave equation
- 2D Poisson equation
- comparison against classical finite-difference solver

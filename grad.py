import torch


def d(y, x):
    # first derivative dy/dx using autograd
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def d2(y, x):
    # second derivative d^2y/dx^2
    return d(d(y, x), x)

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Small local Lion implementation to avoid adding a network dependency."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay:
                    param.data.mul_(1.0 - lr * weight_decay)
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                param.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper around an existing optimizer class."""

    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = {"rho": rho, "adaptive": adaptive, **kwargs}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                e_w = (torch.pow(param, 2) if group["adaptive"] else 1.0) * param.grad * scale.to(param)
                param.add_(e_w)
                self.state[param]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.sub_(self.state[param].pop("e_w"))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires first_step/second_step or a closure")
        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                scale = torch.abs(param) if group["adaptive"] else 1.0
                norms.append((scale * param.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

import torch
from torch.optim import Optimizer
import math 


class adamw(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr":lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            m = 0
            v = 0
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                
                #print(f"step: {state['step']}")
                alpha = lr * math.sqrt(1 - beta2**(state["step"] + 1)) / (1 - beta1**(state["step"] + 1))
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                p.data -= alpha * m / (torch.sqrt(v) + eps)
                state["step"] += 1
                state["m"] = m
                state["v"] = v
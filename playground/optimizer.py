# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:35:07 2025

@author: 5914787
"""
import numpy as np
import torch


class Optimizer:
    def __init__(self, F: np.ndarray, device="cuda"):
        self.F = F
        self.device = device
        self.Ft = torch.from_numpy(F).to(torch.complex128).device(device)
        self.nparams, self.nhilbert, _ = F.shape

    def J(self, E, engine=torch):
        return engine.einsum("mnij, ...i, ...j -> ...mn", self.Ft, E.conj(), E).real

    def normalize(self, E):
        return E / torch.linalg.norm(E)

    def torch_optimize_loss(self, loss_function, steps=2000, lr=1e-3, verbose=True):
        E_param = torch.normal(
            0,
            1,
            (self.nhilbert,),
            dtype=torch.complex128,
            device=self.device,
            requires_grad=True,
        )
        with torch.no_grad():
            E_param = self.normalize(E_param)

        opt = torch.optim.Adam([E_param], lr=lr)
        best_val = np.inf
        best_vec = None
        loss_history = []

        for step in range(steps):
            opt.zero_grad()

            # Perform "soft" normalization. This impacts the gradient and makes
            # sure the gradient does not point in the radial direction: it is
            # tangent to the sphere.
            E_norm = self.normalize(E_param)

            # Calculate the Fisher information matrix and pass it to the loss
            # function
            J = self.J(E_norm)
            loss = loss_function(J)
            loss.backward()
            opt.step()

            # Perform "hard" normalization. This makes sure the input stays
            # normalized, even with numerical errors.
            with torch.no_grad():
                E_param = self.normalize(E_param)

            cur_val = loss.item()
            if cur_val < best_val:
                best_val = cur_val
                best_vec = E_param.detach().cpu().numpy().copy()

            loss_history.append(cur_val)

            if verbose and step % 10 == 0:
                print(f"\rStep {step:4d} => loss={cur_val:.3f}", end="")

        return {"bestval": best_val, "bestvec": best_vec, "loss": loss_history}

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:35:07 2025

@author: 5914787
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


class Optimizer:
    """
    Optimizer class that revolves around the Fisher matrix F. Can do both
    eigenvalue optimizations and torch based optimizations. eig_optimize
    expects a function that takes in F and returns a square tensor.
    torch_optimize expects a functions that takes in J as a torch tensor and
    returns a scalar.
    """

    def __init__(self, F: np.ndarray, device="cuda"):
        self.F = F
        self.Ft = None
        self.device = device

        self.nparams, self.nhilbert, _ = F.shape

    def eig_optimize(self, tensor_function, maximize=True, num=1):
        """
        Optimizes as an eigenvalue problem.

        Parameters
        ----------
        tensor_function: function
            Function that expects the Fisher matrix with shape
            (nparams, nparams, nhilbert, nhilbert) and returns an array of shape
            (nhilbert, nhilbert).
        maximize: bool. Defaults to True.
            Decides whether to return the largest or smallest eigenvalues /
            eigenvectors.
        num: integer or None. Defaults to 1.
            Returns the largest (smallest) num eigenvalues and eigenvectors

        Returns
        -------
        A tuple (eigvals, eigvecs)

        eigvals: (nhilbert,) array
            Largest (smallest) num eigenvalues sorted in descending (ascending)
            order if maximize is True (False)
        eigvecs: (nhilbert, nhilbert) array

        """
        T = tensor_function(self.F)
        eigvals, eigvecs = np.linalg.eig(T)

        sorted_index = eigvals.argsort()

        if maximize:
            sorted_index = np.flip(sorted_index)

        if num == 1:
            sorted_index = sorted_index[0]  # Return single element
        else:
            sorted_index = sorted_index[:num]  # Return slice of first num

        eigvals = eigvals[sorted_index]
        eigvecs = eigvecs[:, sorted_index]

        return eigvals, eigvecs

    def J(self, E):
        return torch.einsum("mnij, ...i, ...j -> ...mn", self.Ft, E.conj(), E).real

    def J_np(self, E):
        return np.einsum("mnij, ...i, ...j -> ...mn", self.F, E.conj(), E).real

    def normalize(self, E):
        return E / torch.linalg.norm(E)

    @staticmethod
    def plot_loss(loss, title=None):
        plt.figure()
        plt.plot(loss)
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.title(title)
        plt.show()

    def torch_optimize_loss_multirun(
        self,
        loss_function,
        steps=2000,
        runs=100,
        lr=1e-3,
        verbose=True,
        maximize=False,
        seed=None,
    ):
        """
        When you try your best but you don't succeed
        Then running it twice is just what you need
        """
        if not seed is None:
            torch.manual_seed(seed)

        if self.Ft is None:
            self.Ft = torch.from_numpy(self.F).to(torch.complex128).device(self.device)

        loss_histories = []

        best_val = -np.inf if maximize else np.inf
        best_vec = None

        for i in range(runs):
            cur_val, cur_vec, loss = self.torch_optimize_loss(
                loss_function, steps=steps, lr=lr, verbose=verbose, maximize=maximize
            ).values()
            if (
                (cur_val < best_val)
                and (not maximize)
                or (cur_val > best_val)
                and maximize
            ):
                best_val = cur_val
                best_vec = cur_vec

            loss_histories.append(loss)

        return {"bestval": best_val, "bestvec": best_vec, "loss": loss_histories}

    def torch_optimize_loss(
        self,
        loss_function,
        steps=2000,
        lr=1e-3,
        verbose=True,
        maximize=False,
        seed=None,
    ):
        if not seed is None:
            torch.manual_seed(seed)

        if self.Ft is None:
            self.Ft = torch.from_numpy(self.F).to(torch.complex128).device(self.device)

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

        opt = torch.optim.Adam([E_param], lr=lr, maximize=maximize)

        best_val = -np.inf if maximize else np.inf
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
            if (
                (cur_val < best_val)
                and (not maximize)
                or (cur_val > best_val)
                and maximize
            ):
                best_val = cur_val
                best_vec = E_param.detach().cpu().numpy().copy()

            loss_history.append(cur_val)

            if verbose and step % 10 == 0:
                print(f"\rStep {step:4d} => loss={cur_val:.3f}", end="")

        return {"bestval": best_val, "bestvec": best_vec, "loss": loss_history}

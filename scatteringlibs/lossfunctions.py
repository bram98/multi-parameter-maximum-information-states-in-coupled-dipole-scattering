# scatteringlibs/lossfunctions.py
from typing import Callable

import numpy as np
import torch

from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator
from scatteringlibs.storable import Storable

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic Loss Functions / Objectives for the full J matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LossFunctionsDict(Storable, dict):
    """
    Helper class to save LossFunction objects in. Behaves exactly like a dict().
    Can store itself using the .save() method from the Storable class. It can
    also select a subset using the .select_only() method.
    """

    def select_only(self, keys):
        selection = {key: value for key, value in self.items() if key in keys}
        return LossFunctionsDict(**selection)

    def names(self):
        return [lf.name for lf in self.values()]

    def displaynames(self):
        return [lf.displayname for lf in self.values()]

    def shortnames(self):
        return [lf.shortname for lf in self.values()]


class LossFunction(Storable):
    """
    Class for handling loss functions conveniently. Encompasses:
        1. A loss function that takes in the Fisher matrix or J and returns a
        scalar
        2.

    Parameters
    ----------
    loss_function : function or lambda expression
        Expects a Fisher matrix F and should return a scalar in case method=='torch'
        and should return a matrix in case method=='eig'.
    name: str
        Name of method. Used in printing.
    displayname: str, optional
        Used in plots. If not provided, `name` will be used.
    shortname: str, optional
        Used as an abbreviation. Can be used in plots along the axis. Not used
        by this class.
    maximize: bool, default: True
        If True the loss_function will be maximized, otherwise minimized.
    method: {'eig', 'torch'}
        For 'eig', the optimized will calculate the largest eigenvalue of the
        matrix returned by loss_function. Otherwise it will use torch
        optimization.
    """

    _optim_methods = ("eig", "torch")
    photon_budget = 1.0

    def __init__(
        self,
        loss_function,
        name,
        displayname=None,
        shortname=None,
        maximize=True,
        method="eig",
        steps=2000,
        lr=1e-3,
        lr_decay=None,
        lr_decay_step_size: int = 500,
        multiruns=None,
    ):
        self.loss_function = loss_function
        self.name = name
        if shortname is None:
            shortname = name
        self.shortname = shortname
        if displayname is None:
            displayname = name
        self.displayname = displayname
        self.maximize = maximize
        if not method in LossFunction._optim_methods:
            raise ValueError(
                f"Method '{method}' not supported. Please use "
                "one of " + str(LossFunction._optim_methods)
            )
        self.method = method

        self._wavefront = None
        self._J = None
        self.bestval = None
        self.loss_history = None

        self.steps = steps
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.multiruns = multiruns

    @staticmethod
    def normalize_wavefront(wavefront):
        """
        Normalizes wavefront. Use the class variable LossFunction.photon_budget
        to determine the final scale.
        """

        norm_sq = np.sum(np.abs(wavefront) ** 2)

        if norm_sq <= 0.0:
            return wavefront

        return wavefront * np.sqrt(LossFunction.photon_budget / norm_sq)

    def optimize(
        self,
        optimizer: Optimizer,
        seed=None,
        steps=None,
        multiruns=None,
        num_eig=1,
        verbose=True,
        lr=None,
        lr_decay=None,
        lr_decay_step_size=None,
        photon_budget=1.0,
    ):
        """
        Performs the optimization. If method == 'eig', the optimizer finds the
        eigenvectors and eigenvalues of the matrix returned by loss_function.
        If method == 'torch', the loss_function is optimized using torch. The
        loss_function should return a scalar and for calculations, torch
        methods should be used.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer from the libraries.optimizer class
        seed: int, default: None
            Provide a seed for torch optimizer to make the run deterministic.
        steps: int, default: None
            Number of steps for torch optimizer. When not given, uses the steps
            attribute, which has default 2000.
        multiruns: int, default: None
            How many times to repeat the torch optimization. When not given,
            uses the instance attribute. When that is also None, the optimizer
            runs once.
        num_eig: int, default: 1
            How many eigenvalues/eigenvectors to return for the eig optimizer
        verbose: bool, default: True
            Enable printing.
        """

        if self.method == "eig":
            bestval, wavefront = optimizer.eig_optimize_F_general(
                self.loss_function, maximize=self.maximize, num=num_eig
            )
            self._wavefront = LossFunction.normalize_wavefront(wavefront)
            self.bestval = bestval

        elif self.method == "torch":
            steps = steps or self.steps
            multiruns = multiruns or self.multiruns
            lr = lr or self.lr
            lr_decay = lr_decay or self.lr_decay
            lr_decay_step_size = lr_decay_step_size or self.lr_decay_step_size

            if multiruns is None:
                result = optimizer.torch_optimize_loss(
                    self.loss_function,
                    steps=steps,
                    lr=lr,
                    lr_decay=lr_decay,
                    lr_decay_step_size=lr_decay_step_size,
                    verbose=verbose,
                    maximize=self.maximize,
                    photon_budget=photon_budget,
                    seed=seed,
                )
                self.loss_history = result["loss_history"]
            else:
                result = optimizer.torch_optimize_loss_multirun(
                    self.loss_function,
                    runs=multiruns,
                    steps=steps,
                    lr=lr,
                    lr_decay=lr_decay,
                    lr_decay_step_size=lr_decay_step_size,
                    verbose=verbose,
                    maximize=self.maximize,
                    photon_budget=photon_budget,
                    base_seed=seed,
                )
                self.loss_history = result["all_loss_histories"]
            self._wavefront = LossFunction.normalize_wavefront(
                result["bestvec"]
            )
            self.bestval = result["bestval"]
            # print(result)

        self._J = optimizer.J_fisher_info_matrix(self._wavefront)

        if verbose:
            print(f"Optimized {self.name: <30}loss: {self.bestval:.5g}")

    def visualize(
        self, sim: CoupledDipoleSimulator, kx_array, title=None, dpi=100
    ):
        """
        Plot wavefront and scatterers.

        Parameters
        ----------
        sim: CoupledDipoleSimulator
            The simulator that is used
        kx_array: np.array
            Input kx_vectors. Can be seen as input angles. Must have same
            dimension as wavefront
        title: str, default: None
            Title to be used on plot
        dpi: int, default: 100
            dpi to be used in plot.

        """
        sim.reset()
        sim.set_input_field_kx(kx_array, self.wavefront)
        sim.solve_fields()

        if title is None:
            title = f"{self.displayname} Optimized Wavefront"

        sim.visualize_intensity(title=title, dpi=dpi)

    @property
    def wavefront(self):
        if self._wavefront is None:
            raise ValueError(
                "Wavefront not defined. Please run optimizer first."
            )

        return self._wavefront

    # @wavefront.setter
    # def wavefront(self, wavefront):
    #     self._wavefront = wavefront

    @property
    def J(self):
        if self._J is None:
            raise ValueError("J is not defined. Please run optimizer first.")
        return self._J

    def __str__(self):
        return f"<LossFunction {self.name}>"

    def __repr__(self):
        return f"<LossFunction {self.name}>"


class LossFunctionSweep(LossFunction):
    def __init__(
        self,
        loss_function,
        name,
        displayname=None,
        shortname=None,
        maximize=True,
        method="eig",
        steps=2000,
        steps_init=2000,
        lr=1e-3,
        lr_decay=None,
        lr_decay_step_size: int = 500,
        multiruns=None,
        E_init_strength: float = 0.1,
        n_sweep: int = 10,
        alpha_min: float = 0,
        alpha_max: float = 1,
    ):
        super().__init__(
            loss_function=loss_function,
            name=name,
            displayname=displayname,
            shortname=shortname,
            maximize=maximize,
            method=method,
            steps=steps,
            lr=lr,
            lr_decay=lr_decay,
            lr_decay_step_size=lr_decay_step_size,
            multiruns=multiruns,
        )

        self.alpha = np.linspace(alpha_min, alpha_max, num=n_sweep)
        self._wavefronts = None
        self._Js = None
        self.steps_init = steps_init
        self.E_init_strength = E_init_strength

    def optimize(
        self,
        optimizer: Optimizer,
        seed: int = None,
        steps: int = None,
        steps_init: int = None,
        multiruns: int = None,
        num_eig: int = 1,
        verbose: bool = True,
        lr: float = None,
        lr_decay: float = None,
        lr_decay_step_size: float = None,
        photon_budget=1.0,
        E_init_strength: float = None,
    ):
        """
        Performs the optimization. If method == 'eig', the optimizer finds the
        eigenvectors and eigenvalues of the matrix returned by loss_function.
        If method == 'torch', the loss_function is optimized using torch. The
        loss_function should return a scalar and for calculations, torch
        methods should be used.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer from the libraries.optimizer class
        seed: int, default: None
            Provide a seed for torch optimizer to make the run deterministic.
        steps: int, default: None
            Number of steps for torch optimizer. When not given, uses the steps
            attribute, which has default 2000.
        multiruns: int, default: None
            How many times to repeat the torch optimization. When not given,
            uses the instance attribute. When that is also None, the optimizer
            runs once.
        num_eig: int, default: 1
            How many eigenvalues/eigenvectors to return for the eig optimizer
        verbose: bool, default: True
            Enable printing.
        """
        self._wavefronts = []

        if verbose:
            print(f"Starting sweep optimization for {self.name}...")

        if self.method == "eig":
            self.bestvals = []

            for alpha in self.alpha:
                bestval, wavefront = optimizer.eig_optimize_F_general(
                    self.loss_function(alpha),
                    maximize=self.maximize,
                    num=num_eig,
                )
                wavefront = LossFunction.normalize_wavefront(wavefront)
                self._wavefronts.append(wavefront)
                self.bestvals.append(bestval)

            self._wavefronts = np.array(self._wavefronts)
            self.bestvals = np.array(self.bestvals)

            if verbose:
                print(
                    f"\rEigenvalue optimization. Alpha={alpha:.3f}/{self.alpha[-1]}"
                )

        elif self.method == "torch":
            steps_init = steps_init or self.steps_init
            steps = steps or self.steps
            multiruns = multiruns or self.multiruns

            optimize_kwargs = dict(
                lr=lr or self.lr,
                lr_decay=lr_decay or self.lr_decay,
                lr_decay_step_size=lr_decay_step_size
                or self.lr_decay_step_size,
                verbose=verbose,
                maximize=self.maximize,
                photon_budget=photon_budget,
                E_init_strength=E_init_strength or self.E_init_strength,
            )
            optimize_multirun_kwargs = optimize_kwargs.copy()
            optimize_kwargs["seed"] = seed
            optimize_multirun_kwargs.update(
                base_seed=seed,
            )

            self._wavefronts = []
            self.loss_history = []
            self.bestvals = []
            # print(multiruns)
            if multiruns is None:
                result = optimizer.torch_optimize_loss(
                    self.loss_function(self.alpha[0]),
                    steps=steps_init,
                    print_prepend=f"Alpha: {self.alpha[0]:.3f}/{self.alpha[-1]}, ",
                    **optimize_kwargs,
                )
                self.loss_history.append(result["loss_history"])
                wavefront_init = result["bestvec"]
                self._wavefronts.append(wavefront_init)

                # print(wavefront_init)
                for alpha in self.alpha[1:]:
                    result = optimizer.torch_optimize_loss(
                        self.loss_function(alpha),
                        steps=steps,
                        print_prepend=f"Alpha: {alpha:.3f}/{self.alpha[-1]}, ",
                        E_init=wavefront_init,
                        **optimize_kwargs,
                    )
                    self.loss_history.append(result["loss_history"])
                    self._wavefronts.append(result["bestvec"])
                    self.bestvals.append(result["bestval"])

            else:
                result = optimizer.torch_optimize_loss_multirun(
                    self.loss_function(self.alpha[0]),
                    steps=steps_init,
                    runs=multiruns,
                    print_prepend=f"Alpha: {self.alpha[0]:.3f}/{self.alpha[-1]}, ",
                    **optimize_multirun_kwargs,
                )
                self.loss_history.append(result["all_loss_histories"])
                wavefront_init = result["bestvec"]
                self._wavefronts.append(wavefront_init)
                self.bestvals.append(result["bestval"])

                for alpha in self.alpha[1:]:
                    result = optimizer.torch_optimize_loss_multirun(
                        self.loss_function(alpha),
                        steps=steps,
                        runs=multiruns,
                        print_prepend=f"Alpha: {alpha:.3f}/{self.alpha[-1]}, ",
                        E_init=wavefront_init,
                        **optimize_multirun_kwargs,
                    )
                    self.loss_history.append(result["all_loss_histories"])
                    self._wavefronts.append(result["bestvec"])
                    self.bestvals.append(result["bestval"])
                    # print(
                    #     f'Optimization sweep alpha={alpha}/{alpha[-1]}  Best loss: {result["bestval"]}'
                    # )

            # self._wavefront = LossFunction.normalize_wavefront(
            #     result["bestvec"]
            # )
            # self.bestval = result["bestval"]
            # print(result)

        self._Js = []
        for wavefront in self._wavefronts:
            self._Js.append(optimizer.J_fisher_info_matrix(wavefront))

        if verbose:
            print(
                f"Optimized {self.name: <30} over {self.n_sweep} interpolations"
            )

    @property
    def n_sweep(self):
        return len(self.alpha)

    @property
    def wavefronts(self):
        if self._wavefronts is None:
            raise ValueError(
                "Wavefronts not defined. Please run optimizer first."
            )

        return self._wavefronts

    @property
    def Js(self):
        if self._Js is None:
            raise ValueError("Js not defined. Please run optimizer first.")
        return self._Js

    @property
    def J(self):
        raise ValueError(
            "J not defined for LossFunctionSweep. Do you mean Js?"
        )

    @property
    def wavefront(self):
        raise ValueError(
            "wavefront not defined for LossFunctionSweep. Do you mean wavefronts?"
        )


def single_param_F(i: int) -> Callable:
    """
    Used for calculating the single parameter Fisher information.
    Args:
        i (int): the parameter to optimize for
    Returns:
        lambda: function that takes in F and returns F[i, i]
    """
    return lambda F: F[i, i]


def single_param_inv_J(i: int) -> Callable:
    """
    Used for calculating the single parameter Fisher information using the
    joint Cramer-Rao lower bound, i.e. by calculating the inverse.
    Args:
        i (int): the parameter to optimize for
    Returns:
        lambda: function that takes in J and returns torch.linalg.inv(J)[i, i]
    """
    return lambda J: torch.linalg.inv(J)[i, i]


def determinant_of_J(J_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the determinant of the Fisher Information Matrix J.
    For D-optimality, this value should be maximized.
    Args:
        J_mat (torch.Tensor): The Fisher Information Matrix (nparams, nparams).
    Returns:
        torch.Tensor: The determinant of J_mat.
    """
    return torch.linalg.det(J_mat)


def trace_of_J(J_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the trace of the Fisher Information Matrix J.
    Args:
        J_mat (torch.Tensor): The Fisher Information Matrix (nparams, nparams).
    Returns:
        torch.Tensor: The trace of J_mat.
    """
    return torch.trace(J_mat)


def trace_of_F(F: np.ndarray) -> np.ndarray:
    """
    Calculates the trace of the Fisher Information operator F.
    Args:
        F (np.ndarray): The Fisher Information Matrix (nparams, nparams, nhilbert, nhilbert).
    Returns:
        np.ndarray: The trace of F with shape (nhilbert, nhilbert).
    """
    return np.trace(F, axis1=0, axis2=1)


def F_weighted(F: np.ndarray, w) -> np.ndarray:
    eig_max = [np.linalg.eigvalsh(F[i, i])[-1] for i in range(3)]
    scale_factors = 1 / np.sqrt(eig_max)
    scale_factors = np.outer(w, w)  # Create matrix
    scale_factors = scale_factors[:, :, np.newaxis, np.newaxis]
    return F * scale_factors


def normalized_trace_of_F(F: np.ndarray) -> np.ndarray:
    eig_max = [np.linalg.eigvalsh(F[i, i])[-1] for i in range(F.shape[0])]
    w = 1 / np.sqrt(eig_max)
    return trace_of_F(F_weighted(F, w))


def trace_of_J_inv(J_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the trace of the inverse of the Fisher Information Matrix J.
    For A-optimality, this value should be minimized.
    Args:
        J_mat (torch.Tensor): The Fisher Information Matrix (nparams, nparams).
    Returns:
        torch.Tensor: The trace of J_mat_inv.
    """
    J_inv = torch.linalg.inv(J_mat)
    return torch.trace(J_inv)


def min_eigenvalue_of_J(J_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the smallest eigenvalue of the Fisher Information Matrix J.
    For E-optimality, this value should be maximized.
    Args:
        J_mat (torch.Tensor): The Fisher Information Matrix (nparams, nparams).
                               Must be Hermitian (real symmetric for FIM).
    Returns:
        torch.Tensor: The smallest eigenvalue of J_mat.
    """
    # eigvalsh returns eigenvalues in ascending order for Hermitian matrices.
    eigvals = torch.linalg.eigvalsh(J_mat)
    return eigvals[0]  # Smallest eigenvalue


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper function for Nuisance Parameter Scenarios
# (Ported from playground/calc_fisher_info_and_MIS_3params.py)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dropped_lossfunction(
    loss_function: Callable, nuisance_param_index: int = 2
) -> Callable:
    return lambda J: loss_function(
        compute_J_interest_drop_nuisance(J, nuisance_param_index)
    )


def partial_lossfunction(
    loss_function: Callable, nuisance_param_index: int = 2
) -> Callable:
    return lambda J: loss_function(
        compute_J_interest_given_nuisance(J, nuisance_param_index)
    )


def compute_J_interest_drop_nuisance(
    J_full: torch.tensor, nuisance_param_index: int = 2
) -> torch.Tensor:
    # Get all indices from 0 to n-1
    all_indices = list(range(J_full.shape[0]))

    # Keep indices that are not the nuisance parameter
    interest_indices = [i for i in all_indices if i != nuisance_param_index]

    # Use advanced indexing to select the rows and columns for the sub-matrix
    J_drop = J_full[
        torch.tensor(interest_indices, device=J_full.device)[:, None],
        torch.tensor(interest_indices, device=J_full.device),
    ]

    return J_drop


def compute_J_interest_given_nuisance(
    J_full: torch.Tensor, nuisance_param_index: int = 2
) -> torch.Tensor:
    """
    Computes the Fisher information matrix for parameters of interest,
    accounting for a single nuisance parameter using Schur complement.
    Assumes J_full is a 3x3 torch tensor (classical FIM).
    The parameters are assumed to be ordered (e.g., x, z, rotation).

    Args:
        J_full (torch.Tensor): The full 3x3 Fisher Information Matrix.
        nuisance_param_index (int): Index of the nuisance parameter (0, 1, or 2).
                                    If 0, params of interest are (1, 2).
                                    If 1, params of interest are (0, 2).
                                    If 2, params of interest are (0, 1).
    Returns:
        torch.Tensor: The 2x2 FIM for the parameters of interest.
    """
    if not (0 <= nuisance_param_index <= 2 and J_full.shape == (3, 3)):
        raise ValueError(
            "J_full must be a 3x3 tensor and nuisance_param_index must be 0, 1, or 2."
        )

    all_indices = [0, 1, 2]
    interest_indices = [i for i in all_indices if i != nuisance_param_index]

    J11 = J_full[
        torch.tensor(interest_indices, device=J_full.device)[:, None],
        torch.tensor(interest_indices, device=J_full.device),
    ]
    J12 = J_full[
        torch.tensor(interest_indices, device=J_full.device),
        nuisance_param_index,
    ].unsqueeze(1)
    J21 = J_full[
        nuisance_param_index,
        torch.tensor(interest_indices, device=J_full.device),
    ].unsqueeze(0)
    J22 = (
        J_full[nuisance_param_index, nuisance_param_index]
        .unsqueeze(0)
        .unsqueeze(0)
    )  # Make it 1x1

    J22_inv = torch.linalg.inv(J22)
    J_interest_nuisance = J11 - J12 @ J22_inv @ J21
    return J_interest_nuisance


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loss Functions / Objectives for a given J_block (e.g., J_interest)
# These can be used with the output of compute_J_interest_given_nuisance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def det_of_J_block(J_block: torch.Tensor) -> torch.Tensor:
    """
    Calculates the determinant of a given Fisher Information Matrix block.
    To be maximized for D-optimality on the block.
    Args:
        J_block (torch.Tensor): A Fisher Information Matrix block.
    Returns:
        torch.Tensor: The determinant of J_block.
    """
    return torch.linalg.det(J_block)


def trace_of_J_block_inv(J_block: torch.Tensor) -> torch.Tensor:
    """
    Calculates the trace of the inverse of a given Fisher Information Matrix block.
    To be minimized for A-optimality on the block.
    Args:
        J_block (torch.Tensor): A Fisher Information Matrix block.
    Returns:
        torch.Tensor: The trace of J_block_inv.
    """
    J_block_inv = torch.linalg.inv(J_block)
    return torch.trace(J_block_inv)


def min_eigenvalue_of_J_block(J_block: torch.Tensor) -> torch.Tensor:
    """
    Calculates the smallest eigenvalue of a given Fisher Information Matrix block.
    To be maximized for E-optimality on the block.
    Args:
        J_block (torch.Tensor): A Fisher Information Matrix block.
                                Must be Hermitian (real symmetric).
    Returns:
        torch.Tensor: The smallest eigenvalue of J_block.
    """
    eigvals = torch.linalg.eigvalsh(J_block)
    return eigvals[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loss Functions for 2x2 Marginal Fisher Information Matrix
# These functions expect a 2x2 PyTorch tensor marginal_J_torch.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def determinant_of_marginal_J(marginal_J_torch: torch.Tensor) -> torch.Tensor:
    """
    Calculates the determinant of the 2x2 marginal Fisher Information Matrix.
    To be maximized for D-optimality.
    Args:
        marginal_J_torch (torch.Tensor): The 2x2 marginal Fisher Information Matrix.
    Returns:
        torch.Tensor: The determinant of marginal_J_torch.
    """
    if marginal_J_torch.shape != (2, 2):
        raise ValueError(
            f"marginal_J_torch must be a 2x2 tensor. Got {marginal_J_torch.shape}"
        )
    return torch.linalg.det(marginal_J_torch)


def minimize_trace_of_marginal_J_inv(
    marginal_J_torch: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the trace of the inverse of the 2x2 marginal Fisher Information Matrix.
    To be minimized for A-optimality.
    A small regularization term is added before inversion for stability.
    Args:
        marginal_J_torch (torch.Tensor): The 2x2 marginal Fisher Information Matrix.
    Returns:
        torch.Tensor: The trace of the regularized inverse of marginal_J_torch.
    """
    if marginal_J_torch.shape != (2, 2):
        raise ValueError(
            f"marginal_J_torch must be a 2x2 tensor. Got {marginal_J_torch.shape}"
        )

    J_reg = marginal_J_torch

    try:
        J_inv = torch.linalg.inv(J_reg)
    except torch.linalg.LinAlgError as e:
        # Fallback to pseudo-inverse if regularized version is still problematic
        print(
            f"Warning: Direct inversion of regularized marginal_J failed: {e}. Using pseudo-inverse."
        )
        J_inv = torch.linalg.pinv(J_reg)

    return torch.trace(J_inv)


def maximize_min_eig_marginal_J(
    marginal_J_torch: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the smallest eigenvalue of the 2x2 marginal Fisher Information Matrix.
    To be maximized for E-optimality.
    Args:
        marginal_J_torch (torch.Tensor): The 2x2 marginal Fisher Information Matrix.
                                         Must be Hermitian (real symmetric).
    Returns:
        torch.Tensor: The smallest eigenvalue of marginal_J_torch.
    """
    if marginal_J_torch.shape != (2, 2):
        raise ValueError(
            f"marginal_J_torch must be a 2x2 tensor. Got {marginal_J_torch.shape}"
        )
    # eigvalsh returns eigenvalues in ascending order for Hermitian matrices.
    eigvals = torch.linalg.eigvalsh(marginal_J_torch)
    return eigvals[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loss Functions / Objectives for Nuisance Parameter Optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def loglin_loss(
    J_full: torch.Tensor,
    nuisance_param_index: int,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Computes log(J_inv_interest[0,0]) + log(J_inv_interest[1,1]) - alpha * log(J_inv_nuisance).
    This corresponds to the "LogLin" strategy.
    Loss should be minimized.

    Args:
        J_full (torch.Tensor): The full Fisher Information Matrix (typically 3x3).
        nuisance_param_index (int): Index of the nuisance parameter.
        alpha (float): Weighting factor for the nuisance parameter term.
        eps (float): Small epsilon for numerical stability.

    Returns:
        torch.Tensor: The computed loss value.
    """
    J_interest = compute_J_interest_given_nuisance(
        J_full, nuisance_param_index
    )

    # Add epsilon to diagonal before inversion for stability
    J_interest_stable = J_interest + eps * torch.eye(
        J_interest.shape[0], device=J_interest.device, dtype=J_interest.dtype
    )
    J_inv_interest = torch.linalg.inv(J_interest_stable)

    J_nuisance_nuisance = J_full[nuisance_param_index, nuisance_param_index]
    # Add epsilon before inversion and log
    J_inv_nuisance = 1.0 / (J_nuisance_nuisance + eps)

    loss = (
        torch.log(J_inv_interest[0, 0] + eps)
        + torch.log(J_inv_interest[1, 1] + eps)
        - alpha * torch.log(J_inv_nuisance + eps)
    )
    return loss


def offdiag_loss(
    J_full: torch.Tensor,
    nuisance_param_index: int,
    beta: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Computes (J_inv_interest[0,0] + J_inv_interest[1,1]) + beta * sum_of_squares_of_off_diagonal_elements_of_J_inv_full.
    This corresponds to the "OffDiag" strategy.
    Loss should be minimized.

    Args:
        J_full (torch.Tensor): The full Fisher Information Matrix (typically 3x3).
        nuisance_param_index (int): Index of the nuisance parameter.
        beta (float): Weighting factor for the off-diagonal term.
        eps (float): Small epsilon for numerical stability.

    Returns:
        torch.Tensor: The computed loss value.
    """
    J_interest = compute_J_interest_given_nuisance(
        J_full, nuisance_param_index
    )

    # Add epsilon to diagonal before inversion for stability
    J_interest_stable = J_interest + eps * torch.eye(
        J_interest.shape[0], device=J_interest.device, dtype=J_interest.dtype
    )
    J_inv_interest = torch.linalg.inv(J_interest_stable)

    # Add epsilon to diagonal before inversion for stability
    J_full_stable = J_full + eps * torch.eye(
        J_full.shape[0], device=J_full.device, dtype=J_full.dtype
    )
    J_inv_full = torch.linalg.inv(J_full_stable)

    # Determine indices for off-diagonal elements based on nuisance_param_index
    # Assuming parameters of interest are always 0 and 1 if nuisance_param_index is 2
    # Or more generally, the two parameters not indexed by nuisance_param_index
    param_indices = [
        i for i in range(J_full.shape[0]) if i != nuisance_param_index
    ]
    if len(param_indices) != 2:
        raise ValueError(
            "This loss function currently assumes exactly two parameters of interest."
        )

    p_idx_0 = param_indices[0]
    p_idx_1 = param_indices[1]

    off_diag_1 = J_inv_full[p_idx_0, nuisance_param_index]
    off_diag_2 = J_inv_full[p_idx_1, nuisance_param_index]
    off_diag_3 = J_inv_full[
        p_idx_0, p_idx_1
    ]  # Coupling between params of interest in the full inverse

    sum_sq_off_diag = off_diag_1**2 + off_diag_2**2 + off_diag_3**2

    # Ensure diagonal elements of J_inv_interest are positive before summing, though typically they are variances.
    # The problem asks for J_inv_interest[0,0] + J_inv_interest[1,1], which are variances and should be positive.
    loss = (
        J_inv_interest[0, 0] + J_inv_interest[1, 1]
    ) + beta * sum_sq_off_diag
    return loss


def subdet_loss(
    J_full: torch.Tensor, nuisance_param_index: int, eps: float = 1e-12
) -> torch.Tensor:
    """
    Computes -log(det(J_interest)).
    This corresponds to the "SubDet" strategy.
    Loss should be minimized (equivalent to maximizing det(J_interest)).

    Args:
        J_full (torch.Tensor): The full Fisher Information Matrix (typically 3x3).
        nuisance_param_index (int): Index of the nuisance parameter.
        eps (float): Small epsilon for numerical stability in log.

    Returns:
        torch.Tensor: The computed loss value.
    """
    J_interest = compute_J_interest_given_nuisance(
        J_full, nuisance_param_index
    )
    det_J_interest = torch.linalg.det(J_interest)

    # Add epsilon before log to ensure argument is positive
    loss = -torch.log(det_J_interest + eps)
    return loss


def maximize_min_eig_J(J_matrix):
    """
    Calculates the smallest eigenvalue of the Fisher Information Matrix J.
    This is intended to be maximized by the optimizer.

    Args:
        J_matrix (torch.Tensor): The Fisher Information Matrix (nparams, nparams).

    Returns:
        torch.Tensor: The smallest eigenvalue of J.
    """
    if J_matrix.ndim != 2 or J_matrix.shape[0] != J_matrix.shape[1]:
        raise ValueError(
            f"J_matrix must be a 2D square matrix. Got {J_matrix.shape}"
        )
    # eigvalsh returns eigenvalues in ascending order for Hermitian matrices
    eigvals = torch.linalg.eigvalsh(J_matrix)
    return eigvals[0]


def minimize_trace_of_J_inv(J_matrix):
    """
    Calculates the trace of the inverse of the Fisher Information Matrix J.
    This is intended to be minimized by the optimizer.
    Attempts direct inversion, falls back to pseudoinverse if direct inversion fails.

    Args:
        J_matrix (torch.Tensor): The Fisher Information Matrix (nparams, nparams).

    Returns:
        torch.Tensor: The trace of J_inv (or J_pinv).
    """
    if J_matrix.ndim != 2 or J_matrix.shape[0] != J_matrix.shape[1]:
        raise ValueError(
            f"J_matrix must be a 2D square matrix. Got {J_matrix.shape}"
        )

    try:
        # Attempt direct inversion first
        J_inv = torch.linalg.inv(J_matrix)
    except torch.linalg.LinAlgError as e:
        # If direct inversion fails, print a warning and use pseudoinverse
        print(
            f"Warning: Direct inversion failed for J_matrix (shape {J_matrix.shape}, device {J_matrix.device}) with error: {e}. Using pseudoinverse."
        )
        # Before taking pseudoinverse, ensure J_matrix is float or complex.
        # If J_matrix might be int/bool, convert it. Assuming it's already float/complex from context.
        if not (
            J_matrix.dtype == torch.float32
            or J_matrix.dtype == torch.float64
            or J_matrix.is_complex()
        ):
            print(
                f"Warning: J_matrix dtype is {J_matrix.dtype}. Converting to float64 for pinv."
            )
            J_matrix_float = J_matrix.to(
                torch.float64
            )  # or appropriate float type
            J_inv = torch.linalg.pinv(J_matrix_float)
        else:
            J_inv = torch.linalg.pinv(J_matrix)

    trace_J_inv = torch.trace(J_inv)
    return trace_J_inv


def sweep_xz_F(alpha):
    # def xz_F(F):
    #     return (1 - alpha) * F[0, 0] + alpha * F[1, 1]
    def xz_F(F):
        return (1 - alpha) * F[0, 0] + alpha * F[1, 1]

    return xz_F


def sweep_xz_inv_J(alpha):
    def xz_inv_J(J):
        J_inv = torch.linalg.inv(J)
        return (1 - alpha) * J_inv[0, 0] + alpha * J_inv[1, 1]

    return xz_inv_J


def sweep_xz_inv_J_eps(eps):
    def _sweep_xz_inv_J_eps(alpha):
        def xz_inv_J(J):
            J_inv = torch.linalg.inv(J)
            return (1 - alpha) * J_inv[0, 0] ** (1 + eps) + alpha * J_inv[
                1, 1
            ] ** (1 + eps)

        return xz_inv_J

    return _sweep_xz_inv_J_eps

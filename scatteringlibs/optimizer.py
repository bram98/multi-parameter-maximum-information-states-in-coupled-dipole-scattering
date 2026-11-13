# scatteringlibs/optimizer.py
import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs.storable import Storable


class Optimizer(Storable):
    """
    Optimizer class that revolves around the Fisher operator tensor F.
    Can do eigenvalue optimizations and torch-based optimizations.
    """

    def __init__(self, F_operator_tensor: np.ndarray, device="cuda"):
        """
        Initializes the Optimizer.

        Args:
            F_operator_tensor (np.ndarray): The Fisher operator tensor,
                                            expected shape (n_params, n_params, N_in, N_in).
            device (str, optional): Device for torch tensors. Defaults to "cuda".
        """
        if (
            F_operator_tensor.ndim != 4
            or F_operator_tensor.shape[0] != F_operator_tensor.shape[1]
            or F_operator_tensor.shape[2] != F_operator_tensor.shape[3]
        ):
            raise ValueError(
                f"F_operator_tensor must have shape (n_params, n_params, N_in, N_in). "
                f"Got {F_operator_tensor.shape}"
            )
        self.F_op = F_operator_tensor
        self.Ft = None  # Torch version, initialized when needed
        self.device = device
        self.n_params, _, self.n_input_modes, _ = F_operator_tensor.shape
        # self.loss_history_torch = (
        #     None  # For storing loss from torch_optimize_loss
        # )

    def _compute_marginal_fisher_info_matrix(
        self, full_J_torch, nuisance_param_index
    ):
        """
        Computes the marginal Fisher Information Matrix (FIM) for parameters of interest,
        excluding a single nuisance parameter.

        The formula used is J_marginal = J11 - J12 @ torch.linalg.inv(J22) @ J21.

        Args:
            full_J_torch (torch.Tensor): The full FIM (e.g., 3x3 for 3 parameters).
            nuisance_param_index (int): The index (0, 1, or 2) of the nuisance parameter.

        Returns:
            torch.Tensor: The marginal FIM (2x2) for the parameters of interest.
        """
        if not (0 <= nuisance_param_index < full_J_torch.shape[0]):
            raise ValueError(
                f"nuisance_param_index must be between 0 and {full_J_torch.shape[0] - 1}"
            )
        if full_J_torch.shape[0] != 3 or full_J_torch.shape[1] != 3:
            # This specific implementation is for a 3x3 matrix
            # For generalization, the index selection logic would need to be more complex.
            raise ValueError(
                f"full_J_torch must be a 3x3 matrix for this implementation. Got {full_J_torch.shape}"
            )

        all_indices = list(range(full_J_torch.shape[0]))
        params_of_interest_indices = [
            i for i in all_indices if i != nuisance_param_index
        ]

        J11 = full_J_torch[
            torch.tensor(params_of_interest_indices)[:, None],
            torch.tensor(params_of_interest_indices),
        ]
        J12 = full_J_torch[
            torch.tensor(params_of_interest_indices)[:, None],
            torch.tensor([nuisance_param_index]),
        ]
        J21 = full_J_torch[
            torch.tensor([nuisance_param_index])[:, None],
            torch.tensor(params_of_interest_indices),
        ]
        J22_scalar = full_J_torch[nuisance_param_index, nuisance_param_index]
        J22 = J22_scalar.reshape(1, 1)

        try:
            inv_J22 = torch.linalg.inv(J22)
        except RuntimeError as e:
            print(
                f"Warning: Inversion of J22 failed: {e}. J22 was: {J22.item()}"
            )  # noqa
            raise e

        J_marginal = J11 - J12 @ inv_J22 @ J21
        return J_marginal

    @staticmethod
    def normalize_wavefront(
        wavefront_coeffs: np.ndarray, photon_budget: float = 1.0
    ):
        """
        Normalizes a wavefront (array of coefficients) to a given photon budget.

        Args:
            wavefront_coeffs (np.ndarray): The input wavefront coefficients.
            photon_budget (float, optional): The total power/photon budget.
                                            Sum of squared magnitudes will equal this.
                                            Defaults to 1.0.

        Returns:
            np.ndarray: The normalized wavefront.
        """
        current_norm_sq = np.sum(np.abs(wavefront_coeffs) ** 2)
        scale_factor = np.sqrt(photon_budget / current_norm_sq)
        return wavefront_coeffs * scale_factor

    def eig_optimize_matrix(
        self, A_matrix: np.ndarray, maximize: bool = True, num: int = 1
    ):
        """
        Performs eigenvalue decomposition on a given matrix A.

        Args:
            A_matrix (np.ndarray): The matrix to diagonalize (e.g., F_mumu for single param MIS).
            maximize (bool, optional): Whether to return eigenvector(s) for largest
                                       or smallest eigenvalues. Defaults to True.
            num (int, optional): Number of top/bottom eigenvectors to return. Defaults to 1.

        Returns:
            tuple: (eigenvalues, eigenvectors)
                   eigenvalues: Selected eigenvalue(s).
                   eigenvectors: Corresponding eigenvector(s) as columns.
                                 If num=1, returns a single vector.
        """
        if A_matrix.ndim != 2 or A_matrix.shape[0] != A_matrix.shape[1]:
            raise ValueError(
                f"Input matrix A must be 2D square. Got {A_matrix.shape}"
            )

        eigvals, eigvecs = np.linalg.eigh(A_matrix)
        sorted_indices = np.argsort(eigvals)

        if maximize:
            sorted_indices = np.flip(sorted_indices)

        if num == 1:
            selected_indices = sorted_indices[0]
        else:
            selected_indices = sorted_indices[:num]

        return eigvals[selected_indices], eigvecs[:, selected_indices]

    def get_single_parameter_mis(
        self, param_names: list = None, photon_budget: float = 1.0
    ):
        """
        Finds the Maximum Information State (MIS) for each parameter individually
        by diagonalizing the F_mumu blocks of the Fisher operator tensor.

        Args:
            param_names (list, optional): List of parameter names. If None, uses indices.
                                          Length must match self.n_params.
            photon_budget (float, optional): Photon budget for normalizing wavefronts.
                                             Defaults to 1.0.

        Returns:
            dict: A dictionary where keys are parameter names (or indices) and
                  values are dictionaries {'eigenvalue': float, 'wavefront': np.ndarray}.
        """
        if param_names is not None and len(param_names) != self.n_params:
            raise ValueError(
                f"Length of param_names ({len(param_names)}) must match "
                f"number of parameters in F_op ({self.n_params})."
            )

        mis_results = {}
        for i in range(self.n_params):
            F_single_param = self.F_op[i, i, :, :]
            lam_max, v_max = self.eig_optimize_matrix(
                F_single_param, maximize=True, num=1
            )

            normalized_v_max = Optimizer.normalize_wavefront(
                v_max, photon_budget
            )

            key = param_names[i] if param_names else i
            mis_results[key] = {
                "eigenvalue": lam_max,
                "wavefront": normalized_v_max,
            }
            print(
                f"Single-param MIS for '{key}': Max Eigenvalue(F_{key}{key}) = {lam_max:.3g}"
            )
        return mis_results

    def get_weighted_trace_mis(self, photon_budget: float = 1.0):
        """
        Computes the MIS for a weighted trace of the diagonal Fisher operators.
        This involves normalizing each F_mumu block by its largest eigenvalue
        and then applying a user-defined weight to each block.

        The operator to be diagonalized is:
        F_weighted_trace = sum(weights[i] * (1/lambda_max_i) * F_op[i,i,:,:])

        Args:
            photon_budget (float, optional): Photon budget for normalizing the final wavefront.
                                             Defaults to 1.0.

        Returns:
            dict: A dictionary containing the result under the key 'weighted_trace',
                  which includes the largest eigenvalue of the combined operator,
                  the corresponding wavefront (MIS), the weights used, and the
                  normalization factors applied.
        """

        # 1. Compute normalization factors (1 / max_eigenvalue of each F_ii)
        normalization_factors = np.zeros(self.n_params)
        for i in range(self.n_params):
            F_ii = self.F_op[i, i, :, :]
            max_eig_val, _ = self.eig_optimize_matrix(
                F_ii, maximize=True, num=1
            )
            if (
                max_eig_val > 1e-12
            ):  # Avoid division by zero or very small numbers
                normalization_factors[i] = 1.0 / max_eig_val
            else:
                normalization_factors[i] = (
                    0.0  # This parameter contributes nothing
                )

        # 2. Compute the weighted trace operator
        F_weighted_trace_op = np.zeros(
            (self.n_input_modes, self.n_input_modes), dtype=complex
        )
        for i in range(self.n_params):
            F_weighted_trace_op += (
                normalization_factors[i] * self.F_op[i, i, :, :]
            )

        # 3. Find the MIS for this combined operator
        lam_max, v_max = self.eig_optimize_matrix(
            F_weighted_trace_op, maximize=True, num=1
        )

        # 4. Normalize the resulting wavefront
        normalized_v_max = Optimizer.normalize_wavefront(v_max, photon_budget)

        # 5. Format and return the result
        result = {
            "weighted_trace": {
                "eigenvalue": lam_max,
                "wavefront": normalized_v_max,
                "normalization_factors": normalization_factors,
            }
        }
        print(f"Weighted Trace MIS: Max Eigenvalue = {lam_max:.3g}")
        return result

    def eig_optimize_F_general(
        self, matrix_function_of_F, maximize=True, num=1
    ):
        """
        Optimizes a general matrix function of the Fisher operator tensor F_op
        via eigenvalue decomposition.

        Args:
            matrix_function_of_F (function): A function that takes the full F_op
                                             (n_params, n_params, N_in, N_in) and
                                             returns a 2D matrix (N_in, N_in)
                                             to be diagonalized.
            maximize (bool, optional): Maximize or minimize eigenvalues. Defaults to True.
            num (int, optional): Number of eigenvectors to return. Defaults to 1.

        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        target_matrix = matrix_function_of_F(self.F_op)
        return self.eig_optimize_matrix(
            target_matrix, maximize=maximize, num=num
        )

    def J_fisher_info_matrix(self, E_wavefront: np.ndarray):
        """
        Calculates the Fisher Information Matrix (FIM) J for a given input wavefront E.
        J_munu = <E| F_munu_op |E>
        Note: This J might be scaled by a factor (e.g., 1/4) compared to some
        definitions of the FIM, depending on the scaling of F_op.

        Args:
            E_wavefront (np.ndarray): The input wavefront coefficients (N_in,).

        Returns:
            np.ndarray: The (n_params, n_params) Fisher Information Matrix.
        """
        E_c = E_wavefront.conj()
        # Einsum: 'mnij,i,j->mn'
        # F_op is (n_params, n_params, N_in, N_in)
        # E_c is (N_in,)
        # E_wavefront is (N_in,)
        # J_mat = np.einsum('mnij,i,j->mn', self.F_op, E_c, E_wavefront)

        # Loop implementation for clarity, einsum can be tricky with complex conjugates and is needlessly hard to read
        J_mat = np.zeros(
            (self.n_params, self.n_params), dtype=float
        )  # FIM is real
        for mu in range(self.n_params):
            for nu in range(self.n_params):
                F_mu_nu_operator = self.F_op[mu, nu, :, :]  # (N_in, N_in)
                # Expectation value: E_c.T @ F_mu_nu_operator @ E_wavefront
                val = np.dot(E_c, np.dot(F_mu_nu_operator, E_wavefront))
                J_mat[mu, nu] = (
                    val.real
                )  # F_mu_nu_operator is Hermitian, so val should be real
        return J_mat

    def plot_torch_loss(
        self,
        loss_history,
        title="Torch Optimization Loss",
        first_step=None,
        ax=None,
    ):

        # loss_history = np.array(loss_history)

        if not hasattr(loss_history[0], "__len__"):
            loss_history = [loss_history]
        # if len(loss_history.shape) == 1:
        #     loss_history = loss_history[np.newaxis, :]

        newfig = ax is None
        if newfig:
            fig, ax = plt.subplots()

        for lh in loss_history:
            lh = np.array(lh)

            steps_array = np.arange(len(lh))
            ax.plot(steps_array[first_step:], lh[first_step:])
        ax.set_xlabel("Optimization Steps")
        ax.set_ylabel("Loss Value")
        # plt.yscale("log")
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        if newfig:
            plt.title(title)
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.show()

    def plot_torch_loss_sweep(
        self,
        loss_history,
        title="Torch Optimization Loss",
        first_step=None,
    ):
        fig, axes = plt.subplots(ncols=2)
        self.plot_torch_loss(
            loss_history[0], first_step=first_step, ax=axes[0]
        )
        loss_history_not_first = np.array(loss_history[1:])
        if len(loss_history_not_first.shape) < 3:
            loss_history_not_first = loss_history_not_first[np.newaxis]

        for loss_history_cur in loss_history_not_first:
            self.plot_torch_loss(
                loss_history_cur,
                first_step=first_step,
                ax=axes[1],
            )
        plt.title(title)
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

    def torch_optimize_loss(
        self,
        loss_function_of_J,  # Expects a function that takes J_mat (n_params, n_params) torch tensor
        steps=2000,
        lr=1e-3,
        verbose=True,
        print_prepend: str = "",
        maximize=False,
        seed=None,
        photon_budget=1.0,
        record_vec_history=False,
        lr_decay: int = None,
        lr_decay_step_size: int = 500,
        E_init: np.ndarray = None,
        E_init_strength: float = 0.1,
        eps_tikonov: float = 1e-8,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        torch_real = dict(dtype=torch.float64, device=self.device)
        torch_complex = dict(dtype=torch.complex128, device=self.device)

        if self.Ft is None:
            self.Ft = torch.from_numpy(self.F_op).to(**torch_complex)
        self.Ft.requires_grad = False  # F_op is fixed

        E = torch.randn(self.n_input_modes, **torch_real) + 1j * torch.randn(
            self.n_input_modes, **torch_real
        )

        # Initial normalization
        current_norm = torch.linalg.norm(E)
        scale = np.sqrt(photon_budget) / current_norm

        E *= scale

        if E_init is not None:
            # Initialize E_param (wavefront coefficients)
            E_init = torch.tensor(E_init, device=self.device)
            E = E_init_strength * E_init + (1 - E_init_strength) * E
            norm = torch.linalg.norm(E)
            scale = np.sqrt(photon_budget) / norm
            E *= scale

        E.requires_grad = True

        photon_budget_sqrt = torch.tensor(np.sqrt(photon_budget), **torch_real)

        opt = torch.optim.Adam([E], lr=lr, maximize=maximize)

        if lr_decay is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, lr_decay_step_size, gamma=lr_decay
            )

        best_val_loss = torch.tensor(
            -np.inf if maximize else np.inf, **torch_real
        )
        best_vec = torch.zeros(self.n_input_modes, **torch_complex)
        loss_history = torch.zeros(steps, **torch_real)
        vec_history = torch.zeros((steps, self.n_input_modes), **torch_complex)

        tikonov_reg = torch.eye(self.n_params, **torch_real) * eps_tikonov

        for step in range(steps):
            opt.zero_grad()

            # Normalized E for calculating J (soft normalization within gradient path)
            # This normalization affects the gradient direction.
            norm = torch.linalg.norm(E)
            E_norm = E * photon_budget_sqrt / norm

            # Calculate J_mat = <E_norm_for_J | F_op | E_norm_for_J>
            J_mat_torch = torch.einsum(
                "mnij, i,j -> mn", self.Ft, E_norm.conj(), E_norm
            ).real

            # Prevents numerical issues for singular J
            J_mat_torch = J_mat_torch + tikonov_reg

            loss = loss_function_of_J(J_mat_torch)  # custom loss function
            loss.backward()
            opt.step()

            if lr_decay is not None:
                scheduler.step()

            # Hard normalization after step (compensates numerical dir)
            with torch.no_grad():
                norm_hard = torch.linalg.norm(E)
                scale_hard = photon_budget_sqrt / norm_hard
                E *= scale_hard

            current_loss = loss.detach()

            loss_history[step] = current_loss

            if maximize:
                improved = current_loss > best_val_loss
            else:
                improved = current_loss < best_val_loss

            if improved:
                best_val_loss = current_loss
                best_vec = E.detach().clone()

            if record_vec_history:
                vec_history[step] = E.detach()

            if verbose and (step + 1) % 50 == 0:
                lr = opt.param_groups[0]["lr"]
                print(
                    f"\r{print_prepend} Step {step + 1:4d}/{steps}, loss={current_loss:.3e}, lr={lr:.3e}           ",
                    end="",
                )

        best_val_loss = best_val_loss.cpu().numpy()
        best_vec = best_vec.cpu().numpy()
        loss_history = loss_history.cpu().numpy()

        if record_vec_history:
            vec_history = vec_history.cpu().numpy()

        result = {
            "bestval": best_val_loss,
            "bestvec": best_vec,
            "loss_history": loss_history,
        }
        if record_vec_history:
            result["vec_history"] = vec_history

        return result

    def torch_optimize_loss_multirun(
        self,
        loss_function_of_J,
        runs=10,
        steps=2000,
        lr=1e-3,
        lr_decay=None,
        lr_decay_step_size=None,
        verbose=True,
        print_prepend: str = "",
        maximize=False,
        photon_budget=1.0,
        base_seed=None,
        record_vec_history=False,
        E_init=None,
        E_init_strength: float = 0.1,
        eps_tikonov: float = 1e-8,
    ):
        """
        When you try your best but you don't succeed
        Then running it twice is just what you need
        """
        overall_best_val = -np.inf if maximize else np.inf
        overall_best_vec = None
        all_loss_histories = []

        if verbose and not print_prepend:
            print(f"\nStarting multi-run optimization ({runs} runs)...")

        best_vals = []
        best_vecs = []
        for i in range(runs):
            current_seed = None
            if base_seed is not None:
                current_seed = base_seed + i

            # if verbose:
            #     print(f"\n--- Run {i+1}/{runs} (seed: {current_seed}) ---")

            run_result = self.torch_optimize_loss(
                loss_function_of_J,
                steps=steps,
                lr=lr,
                lr_decay=lr_decay,
                lr_decay_step_size=lr_decay_step_size,
                verbose=verbose,  # Let sub-function handle detailed print for its steps
                print_prepend=f"{print_prepend} Run {i+1}/{runs}, Best {overall_best_val:.3e}",
                maximize=maximize,
                seed=current_seed,
                photon_budget=photon_budget,
                record_vec_history=record_vec_history,
                E_init=E_init,
                E_init_strength=E_init_strength,
                eps_tikonov=eps_tikonov,
            )

            all_loss_histories.append(run_result["loss_history"])
            current_best_val_run = run_result["bestval"]

            best_vals.append(run_result["bestval"])
            best_vecs.append(run_result["bestvec"])

            if not np.isnan(current_best_val_run):
                if maximize:
                    if current_best_val_run > overall_best_val:
                        overall_best_val = current_best_val_run
                        overall_best_vec = run_result["bestvec"]
                else:  # minimize
                    if current_best_val_run < overall_best_val:
                        overall_best_val = current_best_val_run
                        overall_best_vec = run_result["bestvec"]
            # if verbose:
            #     print(f"Run {i+1} best value: {current_best_val_run:.3e}")

        if verbose and not print_prepend:
            print(
                f"\nMulti-run optimization complete. Overall best value: {overall_best_val:.3e}"
            )

        best_vals = np.array(best_vals)
        best_vecs = np.array(best_vecs)
        all_loss_histories = np.array(all_loss_histories)

        return {
            "bestval": overall_best_val,
            "bestvec": overall_best_vec,
            "bestvals": best_vals,
            "bestvecs": best_vecs,
            "all_loss_histories": all_loss_histories,
        }

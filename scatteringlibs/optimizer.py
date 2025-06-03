# scatteringlibs/optimizer.py
import matplotlib.pyplot as plt
import numpy as np
import torch


class Optimizer:
    """
    Optimizer class that revolves around the Fisher operator tensor F.
    Can do eigenvalue optimizations and torch-based optimizations.
    """

    def __init__(self, F_operator_tensor: np.ndarray, device="cuda"):
        """
        Initializes the Optimizer.

        Args:
            F_operator_tensor (np.ndarray): The Fisher operator tensor,
                                            expected shape (nparams, nparams, N_in, N_in).
            device (str, optional): Device for torch tensors. Defaults to "cuda".
        """
        if (
            F_operator_tensor.ndim != 4
            or F_operator_tensor.shape[0] != F_operator_tensor.shape[1]
            or F_operator_tensor.shape[2] != F_operator_tensor.shape[3]
        ):
            raise ValueError(
                f"F_operator_tensor must have shape (nparams, nparams, N_in, N_in). "
                f"Got {F_operator_tensor.shape}"
            )
        self.F_op = F_operator_tensor
        self.Ft = None  # Torch version, initialized when needed
        self.device = device
        self.nparams, _, self.n_input_modes, _ = F_operator_tensor.shape
        self.loss_history_torch = (
            None  # For storing loss from torch_optimize_loss
        )

    def _compute_marginal_fisher_info_matrix(self, full_J_torch, nuisance_param_index):
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
        # J22 is a submatrix for nuisance parameters. If one nuisance param, it's a 1x1 matrix.
        J22_scalar = full_J_torch[nuisance_param_index, nuisance_param_index]
        J22 = J22_scalar.reshape(1, 1) # Ensure J22 is a 2D tensor for linalg.inv

        # Add epsilon for numerical stability, similar to what might be done elsewhere
        epsilon = 1e-12 * torch.eye(
            J22.shape[0], device=J22.device, dtype=J22.dtype
        )
        J22_reg = J22 + epsilon

        try:
            inv_J22 = torch.linalg.inv(J22_reg)
        except RuntimeError as e:
            # Handle cases where J22_reg might still be singular or near-singular
            # This could happen if J22 was exactly zero and epsilon is too small
            # Or if J22 is extremely ill-conditioned.
            # For a 1x1 matrix, inv(X) = 1/X. If X is near zero, 1/X is huge.
            # A more robust solution might involve pseudo-inverse or checking condition number.
            print(f"Warning: Inversion of J22 failed: {e}. J22 was: {J22.item()}, J22_reg was: {J22_reg.item()}") # noqa
            # Fallback: if J22 is ~0, then J12 @ inv(J22) @ J21 might be problematic.
            # If J22 is very small, its inverse will be very large.
            # Depending on the physics, if J22 is zero, it might mean the nuisance parameter
            # is not coupled, or the measurement provides no info on it.
            # For now, re-raise or return an error matrix.
            # Or, if J22 is effectively zero, the term J12 @ inv(J22) @ J21 might be considered zero
            # if J12 or J21 are also zero, or if the problem implies decoupling.
            # This is a simplification; true handling depends on context.
            # For a 1x1 J22, if J22 is very small, inv_J22 will be very large.
            if torch.abs(J22_reg.item()) < 1e-20: # If J22 is effectively zero
                 # The term J12 @ inv_J22 @ J21 becomes ill-defined or blows up.
                 # If J22 is truly zero, it implies the nuisance parameter has no variance
                 # or no information content on its own.
                 # In such a case, the marginal might not be well-defined by this formula without context.
                 # One approach: treat the term as zero if J22 is zero, assuming decoupling.
                 # This is a strong assumption.
                 # For now, let's return J11, implying the nuisance part contributed nothing or was ill-defined.
                 # Or, raise an error, as a very large correction term might not be meaningful.
                 print(f"Warning: J22_reg is very close to zero ({J22_reg.item()}). The term J12 @ inv(J22) @ J21 might be ill-defined.") # noqa
                 # Fallback to J11 if J22 is near zero, effectively ignoring the Schur complement term.
                 # This is a pragmatic choice but may need domain-specific justification.
                 # A large penalty might be another option: J_marginal = J11 - very_large_number
                 # Or, if J12 or J21 are also zero, the term is zero.
                 if torch.all(torch.abs(J12) < 1e-20) or torch.all(torch.abs(J21) < 1e-20):
                     inv_J22_times_J21 = torch.zeros_like(J21, dtype=full_J_torch.dtype)
                 else:
                     # If J22 is near zero but J12/J21 are not, this term explodes.
                     # This indicates a problem with the FIM or the parameterization.
                     # Returning J11 is a way to say "we can't properly marginalize".
                     print("Warning: J22 near zero but J12/J21 non-zero. Marginal FIM may be unstable. Returning J11.") # noqa
                     return J11 # Or raise an error
            else:
                # If not near zero but still failed, re-raise
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
                                          Length must match self.nparams.
            photon_budget (float, optional): Photon budget for normalizing wavefronts.
                                             Defaults to 1.0.

        Returns:
            dict: A dictionary where keys are parameter names (or indices) and
                  values are dictionaries {'eigenvalue': float, 'wavefront': np.ndarray}.
        """
        if param_names is not None and len(param_names) != self.nparams:
            raise ValueError(
                f"Length of param_names ({len(param_names)}) must match "
                f"number of parameters in F_op ({self.nparams})."
            )

        mis_results = {}
        for i in range(self.nparams):
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

    def eig_optimize_F_general(
        self, matrix_function_of_F, maximize=True, num=1
    ):
        """
        Optimizes a general matrix function of the Fisher operator tensor F_op
        via eigenvalue decomposition.

        Args:
            matrix_function_of_F (function): A function that takes the full F_op
                                             (nparams, nparams, N_in, N_in) and
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
            np.ndarray: The (nparams, nparams) Fisher Information Matrix.
        """
        E_c = E_wavefront.conj()
        # Einsum: 'mnij,i,j->mn'
        # F_op is (nparams, nparams, N_in, N_in)
        # E_c is (N_in,)
        # E_wavefront is (N_in,)
        # J_mat = np.einsum('mnij,i,j->mn', self.F_op, E_c, E_wavefront)

        # Loop implementation for clarity, einsum can be tricky with complex conjugates and is needlessly hard to read
        J_mat = np.zeros(
            (self.nparams, self.nparams), dtype=float
        )  # FIM is real
        for mu in range(self.nparams):
            for nu in range(self.nparams):
                F_mu_nu_operator = self.F_op[mu, nu, :, :]  # (N_in, N_in)
                # Expectation value: E_c.T @ F_mu_nu_operator @ E_wavefront
                val = np.dot(E_c, np.dot(F_mu_nu_operator, E_wavefront))
                J_mat[mu, nu] = (
                    val.real
                )  # F_mu_nu_operator is Hermitian, so val should be real
        return J_mat

    def plot_torch_loss(
        self,
        loss_history=None,
        title="Torch Optimization Loss",
        first_step=None,
    ):
        if loss_history is None:
            if self.loss_history_torch is None:
                print("No loss history available to plot.")
                return
            loss_history = self.loss_history_torch

        steps_array = np.arange(len(loss_history))
        plot_loss = np.array(loss_history)

        if first_step is not None:
            steps_array = steps_array[first_step:]
            plot_loss = plot_loss[first_step:]

        plt.figure()
        plt.plot(steps_array, np.abs(plot_loss))
        plt.xlabel("Optimization Steps")
        plt.ylabel("Loss Value")
        plt.yscale("log")
        plt.title(title)
        plt.grid(True, linestyle=":")
        plt.show()

    def torch_optimize_loss(
        self,
        loss_function_of_J,  # Expects a function that takes J_mat (nparams, nparams) torch tensor
        steps=2000,
        lr=1e-3,
        verbose=True,
        maximize=False,
        seed=None,
        photon_budget=1.0,
        record_vec_history=False,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        if self.Ft is None:
            self.Ft = torch.from_numpy(self.F_op).to(
                dtype=torch.complex128, device=self.device
            )
        self.Ft.requires_grad = False  # F_op is fixed

        # Initialize E_param (wavefront coefficients)
        E_param_real = torch.randn(
            self.n_input_modes, dtype=torch.float64, device=self.device
        )
        E_param_imag = torch.randn(
            self.n_input_modes, dtype=torch.float64, device=self.device
        )
        E_param_real.requires_grad = True
        E_param_imag.requires_grad = True

        current_E_complex = torch.complex(E_param_real, E_param_imag)
        with torch.no_grad():
            # Initial normalization
            current_norm_sq = torch.sum(torch.abs(current_E_complex) ** 2)
            scale = torch.sqrt(
                torch.tensor(
                    photon_budget, dtype=torch.float64, device=self.device
                )
                / current_norm_sq
            )
            E_param_real *= scale
            E_param_imag *= scale

        opt = torch.optim.Adam(
            [E_param_real, E_param_imag], lr=lr, maximize=maximize
        )

        best_val_loss = -np.inf if maximize else np.inf
        best_vec_coeffs = None
        run_loss_history = []
        vec_history_list = [] if record_vec_history else None

        for step in range(steps):
            opt.zero_grad()

            E_coeffs_complex = torch.complex(E_param_real, E_param_imag)

            # Normalized E for calculating J (soft normalization within gradient path)
            # This normalization affects the gradient direction.
            current_norm_sq_E = torch.sum(torch.abs(E_coeffs_complex) ** 2)
            E_norm_for_J = (
                torch.sqrt(
                    torch.tensor(
                        photon_budget, dtype=torch.float64, device=self.device
                    )
                    / current_norm_sq_E
                )
                * E_coeffs_complex
            )

            # Calculate J_mat = <E_norm_for_J | F_op | E_norm_for_J>
            J_mat_torch = torch.einsum(
                "mnij,i,j->mn", self.Ft, E_norm_for_J.conj(), E_norm_for_J
            ).real

            loss = loss_function_of_J(J_mat_torch)  # custom loss function
            loss.backward()
            opt.step()

            # Hard normalization after step (project back to sphere)
            with torch.no_grad():
                E_current_complex_no_grad = torch.complex(
                    E_param_real, E_param_imag
                )
                final_norm_sq = torch.sum(
                    torch.abs(E_current_complex_no_grad) ** 2
                )
                scale_final = torch.sqrt(
                    torch.tensor(
                        photon_budget, dtype=torch.float64, device=self.device
                    )
                    / final_norm_sq
                )
                E_param_real *= scale_final
                E_param_imag *= scale_final

            current_loss_val = (
                loss.item()
                if not (torch.isnan(loss) or torch.isinf(loss))
                else (float("nan"))
            )
            run_loss_history.append(current_loss_val)

            if record_vec_history:
                # Store the normalized version of E_param
                E_to_store_complex = torch.complex(
                    E_param_real.detach(), E_param_imag.detach()
                )
                vec_history_list.append(
                    E_to_store_complex.cpu().numpy().copy()
                )

            if not np.isnan(current_loss_val):
                if maximize:
                    if current_loss_val > best_val_loss:
                        best_val_loss = current_loss_val
                        best_vec_coeffs = (
                            torch.complex(
                                E_param_real.detach(), E_param_imag.detach()
                            )
                            .cpu()
                            .numpy()
                            .copy()
                        )
                else:
                    if current_loss_val < best_val_loss:
                        best_val_loss = current_loss_val
                        best_vec_coeffs = (
                            torch.complex(
                                E_param_real.detach(), E_param_imag.detach()
                            )
                            .cpu()
                            .numpy()
                            .copy()
                        )

            if verbose and step % 50 == 0:
                print(
                    f"\rStep {step:4d}/{steps} => loss={current_loss_val:.3e}",
                    end="",
                )

        if verbose:
            print(f"\nFinished optimization. Best loss: {best_val_loss:.3e}")

        self.loss_history_torch = run_loss_history

        result = {
            "bestval": best_val_loss,
            "bestvec": (
                Optimizer.normalize_wavefront(best_vec_coeffs, photon_budget)
                if best_vec_coeffs is not None
                else None
            ),
            "loss_history": run_loss_history,
        }
        if record_vec_history:
            result["vec_history"] = vec_history_list

        return result

    def torch_optimize_loss_multirun(
        self,
        loss_function_of_J,
        runs=10,
        steps=2000,
        lr=1e-3,
        verbose=True,
        maximize=False,
        photon_budget=1.0,
        base_seed=None,
    ):
        """
        When you try your best but you don't succeed
        Then running it twice is just what you need
        """
        overall_best_val = -np.inf if maximize else np.inf
        overall_best_vec = None
        all_loss_histories = []

        print(f"Starting multi-run optimization ({runs} runs)...")

        for i in range(runs):
            current_seed = None
            if base_seed is not None:
                current_seed = base_seed + i

            if verbose:
                print(f"\n--- Run {i+1}/{runs} (seed: {current_seed}) ---")

            run_result = self.torch_optimize_loss(
                loss_function_of_J,
                steps=steps,
                lr=lr,
                verbose=verbose,  # Let sub-function handle detailed print for its steps
                maximize=maximize,
                seed=current_seed,
                photon_budget=photon_budget,
                record_vec_history=False,  # Typically not needed for multi-run summary
            )

            all_loss_histories.append(run_result["loss_history"])
            current_best_val_run = run_result["bestval"]

            if not np.isnan(current_best_val_run):
                if maximize:
                    if current_best_val_run > overall_best_val:
                        overall_best_val = current_best_val_run
                        overall_best_vec = run_result["bestvec"]
                else:  # minimize
                    if current_best_val_run < overall_best_val:
                        overall_best_val = current_best_val_run
                        overall_best_vec = run_result["bestvec"]
            if verbose:
                print(f"Run {i+1} best value: {current_best_val_run:.3e}")

        print(
            f"\nMulti-run optimization complete. Overall best value: {overall_best_val:.3e}"
        )
        return {
            "bestval": overall_best_val,
            "bestvec": overall_best_vec,  # Already normalized by torch_optimize_loss
            "all_loss_histories": all_loss_histories,
        }

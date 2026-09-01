import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionSweep
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.storable import Storable


class TimeSharingOptimizer(Optimizer):

    def __init__(
        self,
        F_operator_tensor: np.ndarray,
        n_exposures: int = 2,
        device="cuda",
    ):
        super().__init__(F_operator_tensor, device)
        self.n_exposures = n_exposures

    def torch_optimize_loss_timesharing(
        self,
        loss_function_of_J,  # Expects a function that takes J_mat (nparams, nparams) torch tensor
        exposure_times,
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
    ):
        if seed is not None:
            torch.manual_seed(seed)

        assert len(exposure_times) == self.n_exposures
        exposure_times = torch.tensor(
            exposure_times,
            device=self.device,
            dtype=torch.float64,
        )

        if self.Ft is None:
            self.Ft = torch.from_numpy(self.F_op).to(
                dtype=torch.complex128, device=self.device
            )
        self.Ft.requires_grad = False  # F_op is fixed

        E_param_real = torch.randn(
            self.n_exposures,
            self.n_input_modes,
            dtype=torch.float64,
            device=self.device,
        )
        E_param_imag = torch.randn(
            self.n_exposures,
            self.n_input_modes,
            dtype=torch.float64,
            device=self.device,
        )

        E_coeffs_complex = torch.complex(E_param_real, E_param_imag)
        # Initial normalization
        current_norms = torch.linalg.norm(E_coeffs_complex, axis=0)
        scale = np.sqrt(photon_budget) / current_norms

        E_coeffs_complex *= scale[torch.newaxis, :]

        E_param_real.requires_grad = True
        E_param_imag.requires_grad = True

        opt = torch.optim.Adam(
            [E_param_real, E_param_imag], lr=lr, maximize=maximize
        )
        if lr_decay is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, lr_decay_step_size, gamma=lr_decay
            )

        best_val_loss = -np.inf if maximize else np.inf
        best_vec_coeffs = None
        run_loss_history = []
        vec_history_list = [] if record_vec_history else None

        for step in range(steps):
            opt.zero_grad()
            E_coeffs_complex = torch.complex(E_param_real, E_param_imag)
            # E_coeffs_complex = torch.complex(E_param_real, E_param_imag)

            # Normalized E for calculating J (soft normalization within gradient path)
            # This normalization affects the gradient direction.
            current_norms_E = torch.linalg.norm(E_coeffs_complex, axis=0)

            scales = np.sqrt(photon_budget) / current_norms_E

            E_norm = E_coeffs_complex * scales[torch.newaxis, :]
            # E_norm_for_J = (
            #     torch.sqrt(
            #         torch.tensor(
            #             photon_budget, dtype=torch.float64, device=self.device
            #         )
            #         / current_norm_sq_E
            #     )
            #     * E_coeffs_complex
            # )

            # Calculate J_mat = <E_norm_for_J | F_op | E_norm_for_J>
            J_mat_torch = torch.einsum(
                "mnij, ti, tj -> tmn", self.Ft, E_norm.conj(), E_norm
            ).real

            J_shared = torch.einsum(
                "tmn, t -> mn", J_mat_torch, exposure_times
            )

            loss = loss_function_of_J(J_shared)  # custom loss function
            loss.backward()
            opt.step()

            if lr_decay is not None:
                scheduler.step()

            # Hard normalization after step (project back to sphere)
            with torch.no_grad():
                current_norms_E = torch.linalg.norm(E_coeffs_complex, axis=0)

                scales = np.sqrt(photon_budget) / current_norms_E

                E_norm = E_coeffs_complex * scales[torch.newaxis, :]

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

            if verbose and (step + 1) % 50 == 0:
                lr = opt.param_groups[0]["lr"]
                print(
                    f"\r{print_prepend} Step {step + 1:4d}/{steps}, loss={current_loss_val:.3e}, lr={lr:.3e}           ",
                    end="",
                )
        # if verbose:
        #     print()

        # self.loss_history_torch = run_loss_history

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

    def torch_optimize_loss_timesharing_multirun(
        self,
        loss_function_of_J,
        exposure_times,
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
        E_init=None,
        E_init_strength: float = 0.1,
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

        for i in range(runs):
            current_seed = None
            if base_seed is not None:
                current_seed = base_seed + i

            # if verbose:
            #     print(f"\n--- Run {i+1}/{runs} (seed: {current_seed}) ---")

            run_result = self.torch_optimize_loss_timesharing(
                loss_function_of_J,
                exposure_times,
                steps=steps,
                lr=lr,
                lr_decay=lr_decay,
                lr_decay_step_size=lr_decay_step_size,
                verbose=verbose,  # Let sub-function handle detailed print for its steps
                print_prepend=f"{print_prepend} Run {i+1}/{runs}, Best {overall_best_val:.3e}",
                maximize=maximize,
                seed=current_seed,
                photon_budget=photon_budget,
                record_vec_history=False,  # Typically not needed for multi-run summary
                E_init=E_init,
                E_init_strength=E_init_strength,
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
            # if verbose:
            #     print(f"Run {i+1} best value: {current_best_val_run:.3e}")

        if verbose and not print_prepend:
            print(
                f"\nMulti-run optimization complete. Overall best value: {overall_best_val:.3e}"
            )

        return {
            "bestval": overall_best_val,
            "bestvec": overall_best_vec,  # Already normalized by torch_optimize_loss
            "all_loss_histories": all_loss_histories,
        }


class LossFunctionTimeSharing(LossFunctionSweep):
    def __init__(
        self,
        loss_function,
        name,
        displayname=None,
        shortname=None,
        maximize=True,
        method="torch",
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
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            n_sweep=n_sweep,
            steps_init=steps_init,
            E_init_strength=E_init_strength,
        )
        self.loss_functions = loss_function
        self.J_arrays = []

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
            raise NotImplementedError(
                "Eig optimization not implemented for timesharing"
            )
            # self.bestvals = []

            # for alpha in self.alpha:
            #     bestval, wavefront = optimizer.eig_optimize_F_general(
            #         self.loss_function(alpha),
            #         maximize=self.maximize,
            #         num=num_eig,
            #     )
            #     wavefront = LossFunction.normalize_wavefront(wavefront)
            #     self._wavefronts.append(wavefront)
            #     self.bestvals.append(bestval)

            # self._wavefronts = np.array(self._wavefronts)
            # self.bestvals = np.array(self.bestvals)

            # if verbose:
            #     print(
            #         f"\rEigenvalue optimization. Alpha={alpha:.3f}/{self.alpha[-1]}"
            #     )

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
                result = optimizer.torch_optimize_loss_timesharing(
                    self.loss_function(self.alpha[0]),
                    [1, 0],
                    steps=steps_init,
                    print_prepend=f"Alpha: {self.alpha[0]:.3f}/{self.alpha[-1]}, ",
                    **optimize_kwargs,
                )
                self.loss_history.append(result["loss_history"])
                wavefront_init = result["bestvec"]
                self._wavefronts.append(wavefront_init)

                # print(wavefront_init)
                for alpha in self.alpha[1:]:
                    result = optimizer.torch_optimize_loss_timesharing(
                        self.loss_function(alpha),
                        [1 - alpha, alpha],
                        steps=steps,
                        print_prepend=f"Alpha: {alpha:.3f}/{self.alpha[-1]}, ",
                        E_init=wavefront_init,
                        **optimize_kwargs,
                    )
                    self.loss_history.append(result["loss_history"])
                    self._wavefronts.append(result["bestvec"])
                    self.bestvals.append(result["bestval"])

            else:
                result = optimizer.torch_optimize_loss_timesharing_multirun(
                    self.loss_function(self.alpha[0]),
                    [1, 0],
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
                    result = optimizer.torch_optimize_loss_timesharing_multirun(
                        self.loss_function(alpha),
                        [1 - alpha, alpha],
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
        print()
        self._Js = []
        self._J_arrays = []
        for alpha, wavefront_list in zip(self.alpha, self._wavefronts):
            # print(wavefront.shape)
            self._J_arrays.append([])
            for wavefront in wavefront_list:
                J_cur = optimizer.J_fisher_info_matrix(wavefront)
                self._J_arrays[-1].append(J_cur)
            self._J_arrays[-1] = np.array(self._J_arrays[-1])
            exposure_times = np.array([1 - alpha, alpha])[
                :, np.newaxis, np.newaxis
            ]
            J_total = np.sum(self._J_arrays[-1] * exposure_times, axis=0)
            print(
                J_total.shape, exposure_times.shape, self._J_arrays[-1].shape
            )
            self._Js.append(J_total)

        if verbose:
            print(
                f"Optimized {self.name: <30} over {self.n_sweep} interpolations"
            )


if __name__ == "__main__":
    fisher_calc = FisherMatrix.load(
        folder="../data/quasirandom_shortrun_dist2-8/"
    )
    tso = TimeSharingOptimizer(fisher_calc.F)
    o = Optimizer(fisher_calc.F)

    lf_J = lambda J: torch.linalg.det(J)

    lf = LossFunctionTimeSharing(
        lf_J, "det_timesharing", steps=200, steps_init=200, n_sweep=5
    )
    lf.optimize(tso)


# result = o.torch_optimize_loss(lf_J)

# plt.plot(result["loss_history"][1000:], label="raw")
# for t in np.linspace(0, 1, num=10):
#     result = tso.torch_optimize_loss_timesharing(lf_J, [1 - t, t])

#     plt.plot(result["loss_history"][1000:], label=f"ts - {t:.1f}")

# plt.legend()

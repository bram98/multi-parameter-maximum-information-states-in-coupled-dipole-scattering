import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionSweep
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.storable import Storable

# %%


def print_norm(msg, E):
    E_np = E.cpu().detach().numpy()
    print(msg, np.linalg.norm(E_np, axis=1))


def print_norm2(msg, Ereal, Eimag):
    Ereal_np = Ereal.cpu().detach().numpy()
    Eimag_np = Eimag.cpu().detach().numpy()
    print(msg, np.linalg.norm(Ereal_np + 1j * Eimag_np, axis=1))


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
        exposure_times.requires_grad = False

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
        current_norms = torch.linalg.norm(E_coeffs_complex, axis=1)
        scales = np.sqrt(photon_budget) / current_norms

        E_param_real *= scales[:, torch.newaxis]
        E_param_imag *= scales[:, torch.newaxis]

        # print_norm("init", E_coeffs_complex)
        # print_norm2("init", E_param_real, E_param_imag)

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

            # print_norm("loc1", E_coeffs_complex)
            # print_norm2("loc1", E_param_real, E_param_imag)

            # Normalized E for calculating J (soft normalization within gradient path)
            # This normalization affects the gradient direction.
            current_norms_E = torch.linalg.norm(E_coeffs_complex, axis=1)

            scales = np.sqrt(photon_budget) / current_norms_E

            E_norm = E_coeffs_complex * scales[:, torch.newaxis]

            # print_norm("soft", E_norm)
            # print_norm2("soft", E_param_real, E_param_imag)

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
                current_norms_E = torch.linalg.norm(E_coeffs_complex, axis=1)

                scales = np.sqrt(photon_budget) / current_norms_E

                E_param_real *= scales[:, torch.newaxis]
                E_param_imag *= scales[:, torch.newaxis]

                # print_norm("hard", E_norm)
                # print_norm2("hard", E_param_real, E_param_imag)

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

        result = {
            "bestval": best_val_loss,
            "bestvec": (
                best_vec_coeffs if best_vec_coeffs is not None else None
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
        seed=None,
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
        n_exposure_times: int = 10,
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
        self.n_exposure_times = n_exposure_times
        t = np.linspace(0, 0.5, num=n_exposure_times, endpoint=True)
        self.exposure_times = np.array([1 - t, t]).T
        # self.exposure_times = np.ones((n_exposure_times, 1))

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
                for alpha in self.alpha:

                    self._wavefronts.append([])
                    self.loss_history.append([])
                    self.bestvals.append([])

                    for exposure_time in self.exposure_times:
                        result = optimizer.torch_optimize_loss_timesharing(
                            self.loss_function(alpha),
                            exposure_time,
                            steps=steps,
                            print_prepend=f"Alpha: {alpha:.3f}/{self.alpha[-1]}, \tExposures={exposure_time} ",
                            E_init=None,
                            **optimize_kwargs,
                        )
                        self.loss_history[-1].append(result["loss_history"])
                        self._wavefronts[-1].append(result["bestvec"])
                        self.bestvals[-1].append(result["bestval"])

            else:
                print("NOT IMPLEMENTED")
                for alpha in self.alpha:

                    self._wavefronts.append([])
                    self.loss_history.append([])
                    self.bestvals.append([])

                    for exposure_time in self.exposure_times:
                        result = optimizer.torch_optimize_loss_timesharing_multirun(
                            self.loss_function(alpha),
                            exposure_time,
                            steps=steps,
                            print_prepend=f"Alpha: {alpha:.3f}/{self.alpha[-1]}, \tExposures={exposure_time} ",
                            E_init=None,
                            runs=multiruns,
                            **optimize_kwargs,
                        )
                        self.loss_history[-1].append(
                            result["all_loss_histories"]
                        )
                        self._wavefronts[-1].append(result["bestvec"])
                        self.bestvals[-1].append(result["bestval"])
                # result = optimizer.torch_optimize_loss_timesharing_multirun(
                #     self.loss_function(self.alpha[0]),
                #     [1, 0],
                #     steps=steps_init,
                #     runs=multiruns,
                #     print_prepend=f"Alpha: {self.alpha[0]:.3f}/{self.alpha[-1]}, ",
                #     **optimize_multirun_kwargs,
                # )
                # self.loss_history.append(result["all_loss_histories"])
                # wavefront_init = result["bestvec"]
                # self._wavefronts.append(wavefront_init)
                # self.bestvals.append(result["bestval"])

                # for alpha in self.alpha[1:]:
                #     result = optimizer.torch_optimize_loss_timesharing_multirun(
                #         self.loss_function(alpha),
                #         [1 - alpha, alpha],
                #         steps=steps,
                #         runs=multiruns,
                #         print_prepend=f"Alpha: {alpha:.3f}/{self.alpha[-1]}, ",
                #         E_init=wavefront_init,
                #         **optimize_multirun_kwargs,
                #     )
                #     self.loss_history.append(result["all_loss_histories"])
                #     self._wavefronts.append(result["bestvec"])
                #     self.bestvals.append(result["bestval"])

        print()

        self._J_arrays = []
        self._Js = []
        for alpha, wavefront_list in zip(self.alpha, self._wavefronts):

            self._J_arrays.append([])
            self._Js.append([])

            for exposure_time, wavefronts_shared in zip(
                self.exposure_times, wavefront_list
            ):
                J_arrays = [
                    optimizer.J_fisher_info_matrix(wavefront)
                    for wavefront in wavefronts_shared
                ]
                # J_cur = optimizer.J_fisher_info_matrix(wavefront)
                self._J_arrays[-1].append(J_arrays)
                J_total = np.einsum("tij, t -> ij", J_arrays, exposure_time)

                self._Js[-1].append(J_total)

        self._J_arrays = np.array(self._J_arrays)
        self._Js = np.array(self._Js)
        self.loss_history = np.array(self.loss_history)
        self._wavefronts = np.array(self._wavefronts)
        self.bestvals = np.array(self.bestvals)

        if verbose:
            print(
                f"Optimized {self.name: <30} over {self.n_sweep} interpolations"
            )


if __name__ == "__main__":
    project_folder = "../data/quasirandom_longrun_dist2-1/"
    fisher_calc = FisherMatrix.load(
        folder="../data/quasirandom_longrun_dist2-1/"
    )
    tso = TimeSharingOptimizer(fisher_calc.F, n_exposures=2)
    o = Optimizer(fisher_calc.F)

    # lf_J = lambda J: torch.linalg.det(J)
    def lf_J(alpha):
        def _lf_J(J):
            J_inv = torch.linalg.inv(J)
            return (1 - alpha) * J_inv[0, 0] + alpha * J_inv[1, 1]

        return _lf_J

    lf = LossFunctionTimeSharing(
        lf_J,
        "det_timesharing",
        steps=100,
        steps_init=100,
        n_sweep=3,
        n_exposure_times=3,
        multiruns=None,
        maximize=False,
    )
    lf.optimize(tso)

# %%

lfmr = LossFunctionTimeSharing(
    lf_J,
    "det_timesharing",
    steps=10000,
    steps_init=10000,
    n_sweep=11,
    n_exposure_times=5,
    multiruns=10,
    maximize=False,
)
lfmr.optimize(tso)

# %%
l = lf.loss_history.shape[-1]
plt.plot(lf.loss_history.reshape((-1, l)).T[2000:, :])
plt.show()

# %%
l = lfmr.loss_history.shape[-1]
plt.plot(lfmr.loss_history.reshape((-1, l)).T[:, :])
plt.show()

# %%
J_inv = np.linalg.inv(lf._Js)
sigma_x = J_inv[:, :, 0, 0]
sigma_z = J_inv[:, :, 1, 1]

plt.plot(sigma_x, sigma_z, label="tiemsharing")
plt.legend()
plt.show()
# plt.plot(sigma_x.T, sigma_z.T)
# plt.show()
# %%
lf_ref = LossFunctionSweep(
    lf_J,
    "det_timesharing_ref",
    steps=10000,
    steps_init=10000,
    n_sweep=11,
    multiruns=10,
    method="torch",
    maximize=False,
)
lf_ref.optimize(o)
lf_ref._Js = np.array(lf_ref._Js)
lf_ref.loss_history = np.array(lf_ref.loss_history)

# %%
l_ref = lf_ref.loss_history.shape[-1]
plt.plot(lf_ref.loss_history.reshape((-1, l_ref)).T[-500:, :])
plt.show()

# %%
J_inv_ref = np.linalg.inv(lf_ref._Js)
sigma_x_ref = J_inv_ref[:, 0, 0]
sigma_z_ref = J_inv_ref[:, 1, 1]

timesharing_labels = [
    f"timesharing {exp_time}" for exp_time in lf.exposure_times
]
plt.plot(sigma_x, sigma_z, label=timesharing_labels)
plt.plot(sigma_x_ref, sigma_z_ref, label="single exp")
plt.xlim(0.0675, 0.08)
plt.ylim(0.0225, 0.04)
plt.legend()
plt.show()


# %%
wf = lf_ref.wavefronts[5]
delta = 0.000001
n = len(wf)

# np.random.seed(10)
dE = np.random.randn(n) + 1j * np.random.rand(n)
dE -= np.vecdot(wf, dE) / np.vecdot(wf, wf) * wf

wf_plus = wf + delta * dE
wf_min = wf - delta * dE
wf_plus /= np.linalg.norm(wf_plus)
wf_min /= np.linalg.norm(wf_min)

J_plus = o.J_fisher_info_matrix(wf_plus)
J_min = o.J_fisher_info_matrix(wf_min)

Sigma_plus = np.linalg.inv(J_plus)
Sigma_min = np.linalg.inv(J_min)
dSigma = (Sigma_plus - Sigma_min) / (2 * delta)

# print((J_plus - J_min)/(2*delta))
print(dSigma)

# %%
lf.save(folder=project_folder, name="timesharing")
lf_ref.save(folder=project_folder, name="timesharing_ref")

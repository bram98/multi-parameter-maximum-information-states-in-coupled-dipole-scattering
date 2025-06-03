# scatteringlibs/lossfunctions.py
import torch

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic Loss Functions / Objectives for the full J matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
    For T-optimality, this value should be maximized.
    Args:
        J_mat (torch.Tensor): The Fisher Information Matrix (nparams, nparams).
    Returns:
        torch.Tensor: The trace of J_mat.
    """
    return torch.trace(J_mat)


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

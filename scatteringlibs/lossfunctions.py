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
    # Add a small epsilon to the diagonal for numerical stability if J_mat is singular.
    # This helps prevent NaN gradients if J_mat is ill-conditioned during optimization.
    # A very small epsilon might be needed depending on typical J_mat scales.
    # Consider making epsilon a parameter or handling higher up if issues persist.
    # For now, direct inversion is attempted, optimizer should ideally avoid singularity.
    # J_inv = torch.linalg.inv(J_mat + 1e-12 * torch.eye(J_mat.shape[0], device=J_mat.device, dtype=J_mat.dtype))
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
    return eigvals[0] # Smallest eigenvalue

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper function for Nuisance Parameter Scenarios
# (Ported from playground/calc_fisher_info_and_MIS_3params.py)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_J_interest_given_nuisance(J_full: torch.Tensor, nuisance_param_index: int = 2) -> torch.Tensor:
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
    if not (0 <= nuisance_param_index <= 2 and J_full.shape == (3,3)):
        raise ValueError("J_full must be a 3x3 tensor and nuisance_param_index must be 0, 1, or 2.")

    all_indices = [0, 1, 2]
    interest_indices = [i for i in all_indices if i != nuisance_param_index]

    J11 = J_full[torch.tensor(interest_indices, device=J_full.device)[:, None], torch.tensor(interest_indices, device=J_full.device)]
    J12 = J_full[torch.tensor(interest_indices, device=J_full.device), nuisance_param_index].unsqueeze(1)
    J21 = J_full[nuisance_param_index, torch.tensor(interest_indices, device=J_full.device)].unsqueeze(0)
    J22 = J_full[nuisance_param_index, nuisance_param_index].unsqueeze(0).unsqueeze(0) # Make it 1x1

    # Regularize J22 inversion for numerical stability
    # Adding a small value to the diagonal of J22 before inversion
    # This is equivalent to Tikhonov regularization for J22
    # The value of eps might need tuning depending on the scale of J matrix elements
    eps = 1e-12
    J22_inv = torch.linalg.inv(J22 + eps * torch.eye(J22.shape[0], device=J22.device, dtype=J22.dtype))

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
    # J_block_inv = torch.linalg.inv(J_block + 1e-12 * torch.eye(J_block.shape[0], device=J_block.device, dtype=J_block.dtype))
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

# __all__ can be defined here to specify the public API if desired
# For example:
# __all__ = [
#     "determinant_of_J", "trace_of_J", "trace_of_J_inv", "min_eigenvalue_of_J",
#     "compute_J_interest_given_nuisance",
#     "det_of_J_block", "trace_of_J_block_inv", "min_eigenvalue_of_J_block"
# ]

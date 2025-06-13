import os
import sys  # For exit codes

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Attempt to import the target function
try:
    from scatteringlibs.crlbanalysis import plot_pareto_frontiers

    print(
        "Successfully imported plot_pareto_frontiers from scatteringlibs.crlbanalysis"
    )
except ImportError as e:
    print(f"Error importing plot_pareto_frontiers: {e}")
    # Attempt to find out which module is missing if it's a chain
    current_exception = e
    while current_exception.__cause__:
        print(f"Caused by: {current_exception.__cause__}")
        current_exception = current_exception.__cause__
    if isinstance(e, ModuleNotFoundError):
        print(f"Missing module was: {e.name}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# 1. Define dummy data
print("1. Defining dummy data...")
optimized_wavefronts_dict = {
    "wf1": np.random.rand(10) + 1j * np.random.rand(10),
    "wf2": np.random.rand(10) + 1j * np.random.rand(10),
}

# Modified F_operator_tensor_np for positive definite J
n_params = 2
n_in = 10
F_operator_tensor_np = np.zeros((n_params, n_params, n_in, n_in))
# Make F_mu_nu_op positive definite for mu == nu to ensure J is likely positive definite
for i in range(n_params):
    # Create a random positive definite matrix: A.T @ A + epsilon*I
    A = np.random.rand(n_in, n_in) + 1j * np.random.rand(n_in, n_in)
    F_operator_tensor_np[i, i, :, :] = np.real(A.conj().T @ A) + 0.1 * np.eye(
        n_in
    )
    # Ensure it's real if J calculation expects real F components directly,
    # or handle complex F if J calculation is E_c @ F @ E.
    # Based on _compute_J_for_wavefront_numpy, val.real is taken, so F can be complex.
    # Let's stick to real positive definite for simplicity here in F_mu_nu_op for diagonal J components.
    # For off-diagonal J components (mu != nu), F_operator_tensor_np[mu, nu, :, :] remains zero.

# Ensure wavefronts have non-zero energy
for wf_name in optimized_wavefronts_dict:
    if np.sum(np.abs(optimized_wavefronts_dict[wf_name]) ** 2) == 0:
        optimized_wavefronts_dict[wf_name] = np.ones(
            n_in
        )  # Replace zero wf with ones


photon_budget_val = 1.0
param_names_list = ["param_x", "param_y"]
params_to_plot_indices = [0, 1]  # Plot param_x vs param_y
wavefront_categories_style = {
    "wf1": {"label": "Category A", "marker": "o", "color": "blue"},
    "wf2": {"label": "Category B", "marker": "s", "color": "red"},
    "default": {"label": "Other", "marker": "x", "color": "grey"},
}
wavefront_labels_latex_map = {"wf1": "WF1", "wf2": "WF2"}
filepath = "test_pareto_plot.png"
dpi = 150
print("Dummy data defined.")

# Get current backend to check if it's non-interactive
initial_backend = plt.get_backend()
print(f"Initial Matplotlib backend: {initial_backend}")

# Ensure a non-interactive backend is used for the test
plt.switch_backend("Agg")
print(f"Switched Matplotlib backend to: {plt.get_backend()}")

# 2. Call crlbanalysis.plot_pareto_frontiers
print(f"\n2. Calling plot_pareto_frontiers (saving to {filepath})...")
try:
    plot_pareto_frontiers(
        optimized_wavefronts_dict=optimized_wavefronts_dict,
        F_operator_tensor_np=F_operator_tensor_np,
        photon_budget_val=photon_budget_val,
        param_names_list=param_names_list,
        params_to_plot_indices=params_to_plot_indices,
        wavefront_categories_style=wavefront_categories_style,
        wavefront_labels_latex_map=wavefront_labels_latex_map,
        filepath=filepath,
        dpi=dpi,
    )
    print("plot_pareto_frontiers executed.")
except Exception as e:
    print(f"Error calling plot_pareto_frontiers: {e}")
    import traceback

    traceback.print_exc()
    # Restore backend and exit if function call fails
    plt.switch_backend(initial_backend)
    print(f"Restored Matplotlib backend to: {initial_backend}")
    sys.exit(1)

# 3. Verify the output image
print("\n3. Verifying output image...")
success = False
try:
    if not os.path.exists(filepath):
        print(f"FAILURE: Image file '{filepath}' was not created.")
    else:
        print(f"Image file '{filepath}' found.")
        try:
            img = Image.open(filepath)
            print(f"Image '{filepath}' opened successfully.")
            width, height = img.size
            if width > 0 and height > 0:
                print(f"Image dimensions: {width}x{height} (Non-zero, OK)")
                file_size = os.path.getsize(filepath)
                min_file_size_kb = 2
                if file_size > min_file_size_kb * 1024:
                    print(
                        f"Image file size: {file_size / 1024:.2f} KB (>{min_file_size_kb}KB, OK)"
                    )
                    success = True
                    print("SUCCESS: Image verification passed.")
                else:
                    print(
                        f"FAILURE: Image file size {file_size / 1024:.2f} KB is not > {min_file_size_kb}KB."
                    )
            else:
                print(
                    f"FAILURE: Image dimensions are zero ({width}x{height})."
                )
        except Exception as e:
            print(
                f"FAILURE: Error opening or validating image '{filepath}': {e}"
            )
            import traceback

            traceback.print_exc()

except Exception as e:
    print(
        f"FAILURE: An unexpected error occurred during image verification: {e}"
    )
    import traceback

    traceback.print_exc()

# 4. Clean up
print("\n4. Cleaning up...")
if os.path.exists(filepath):
    try:
        os.remove(filepath)
        print(f"File '{filepath}' deleted.")
    except Exception as e:
        print(f"Error deleting file '{filepath}': {e}")
else:
    print(
        f"File '{filepath}' not found for deletion (may not have been created)."
    )

# Restore initial backend
plt.switch_backend(initial_backend)
print(f"Restored Matplotlib backend to: {initial_backend}")

if success:
    print("\nOverall Test Result: SUCCESS")
else:
    print("\nOverall Test Result: FAILURE")

# Ensure script exits with appropriate code
sys.exit(0 if success else 1)

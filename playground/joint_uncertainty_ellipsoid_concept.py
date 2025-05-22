import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# --- Controllable Plotting Parameters ---
annotation_fontsize = 13  # Slightly larger font size for annotations
axis_label_fontsize = 17  # Font size for axis labels
textbox_line_spacing_factor = (
    0.11  # Smaller value for closer stacking (e.g., 0.1 to 0.15)
)
ellipsoid_alpha = 0.2  # Transparency of the ellipsoid
ellipsoid_edge_lw = 0.5  # Linewidth of ellipsoid edges

# --- Hypothetical Covariance Matrix (Inverse FIM: J^-1) ---
# This matrix represents the CRLBs.
cov_matrix = np.array(
    [
        [0.02, 0.01, 0.005],  # Var(dx), Cov(dx,dz), Cov(dx,dtheta)
        [0.01, 0.015, 0.008],  # Cov(dz,dx), Var(dz), Cov(dz,dtheta)
        [0.005, 0.008, 0.025],  # Cov(dtheta,dx), Cov(dtheta,dz), Var(dtheta)
    ]
)
# Ensure symmetry
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# --- Calculate Ellipsoid Properties ---
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # J_inv = V Lambda V^T

# Sort eigenvalues and eigenvectors: largest eigenvalue of J_inv first
# (corresponds to longest semi-axis of ellipsoid)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues_J_inv = eigenvalues[sorted_indices]
sorted_radii_ellipsoid = np.sqrt(sorted_eigenvalues_J_inv)  # Semi-axis lengths
sorted_eigenvectors_J_inv = eigenvectors[:, sorted_indices]

# --- Generate Ellipsoid Points ---
u = np.linspace(0, 2 * np.pi, 70)
v = np.linspace(0, np.pi, 35)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones_like(u), np.cos(v))

ellipsoid_transformed_points = (
    sorted_eigenvectors_J_inv
    @ np.diag(sorted_radii_ellipsoid)
    @ np.array([x_sphere.ravel(), y_sphere.ravel(), z_sphere.ravel()])
)
x_ellipsoid = ellipsoid_transformed_points[0, :].reshape(x_sphere.shape)
y_ellipsoid = ellipsoid_transformed_points[1, :].reshape(y_sphere.shape)
z_ellipsoid = ellipsoid_transformed_points[2, :].reshape(z_sphere.shape)

# --- Plotting ---
fig = plt.figure(figsize=(10, 8.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for true shape

# Plot the ellipsoid surface
ax.plot_surface(
    x_ellipsoid,
    y_ellipsoid,
    z_ellipsoid,
    rstride=2,
    cstride=2,
    color="deepskyblue",
    alpha=ellipsoid_alpha,
    linewidth=ellipsoid_edge_lw,
    edgecolor="darkblue",
    shade=True,
)

# Plot principal axes (semi-axes)
origin = np.array([0.0, 0.0, 0.0])
axis_linewidth = 2.2

# Longest Semi-Axis (E-optimality related) - The one E-optimality TRIES TO MINIMIZE
longest_axis_vector_end = (
    sorted_eigenvectors_J_inv[:, 0] * sorted_radii_ellipsoid[0]
)
ax.plot(
    [0, longest_axis_vector_end[0]],
    [0, longest_axis_vector_end[1]],
    [0, longest_axis_vector_end[2]],
    color="red",
    linewidth=axis_linewidth,
    linestyle="-",
)
ax.plot(
    [0, -longest_axis_vector_end[0]],
    [0, -longest_axis_vector_end[1]],
    [0, -longest_axis_vector_end[2]],
    color="red",
    linewidth=axis_linewidth,
    linestyle="-",
)

# Second Longest Semi-Axis
intermediate_axis_vector_end = (
    sorted_eigenvectors_J_inv[:, 1] * sorted_radii_ellipsoid[1]
)
ax.plot(
    [0, intermediate_axis_vector_end[0]],
    [0, intermediate_axis_vector_end[1]],
    [0, intermediate_axis_vector_end[2]],
    color="darkmagenta",
    linewidth=axis_linewidth * 0.7,
    linestyle="--",
)
ax.plot(
    [0, -intermediate_axis_vector_end[0]],
    [0, -intermediate_axis_vector_end[1]],
    [0, -intermediate_axis_vector_end[2]],
    color="darkmagenta",
    linewidth=axis_linewidth * 0.7,
    linestyle="--",
)

# Shortest Semi-Axis
shortest_axis_vector_end = (
    sorted_eigenvectors_J_inv[:, 2] * sorted_radii_ellipsoid[2]
)
ax.plot(
    [0, shortest_axis_vector_end[0]],
    [0, shortest_axis_vector_end[1]],
    [0, shortest_axis_vector_end[2]],
    color="forestgreen",
    linewidth=axis_linewidth * 0.7,
    linestyle="--",
)
ax.plot(
    [0, -shortest_axis_vector_end[0]],
    [0, -shortest_axis_vector_end[1]],
    [0, -shortest_axis_vector_end[2]],
    color="forestgreen",
    linewidth=axis_linewidth * 0.7,
    linestyle="--",
)

# Annotations for Optimality Criteria (all stacked top-left)
text_props = dict(
    boxstyle="round,pad=0.3", fc="lightgoldenrodyellow", alpha=0.85
)
v_offset = 0.98
h_pos = 0.02

ax.text2D(
    h_pos,
    v_offset,
    r"$\bf{D-Opt:}$ Min($\det(J^{-1})$)" "\n  (Minimize Ellipsoid Volume)",
    transform=ax.transAxes,
    fontsize=annotation_fontsize,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=text_props,
)
v_offset -= textbox_line_spacing_factor - 0.01

ax.text2D(
    h_pos,
    v_offset,
    r"$\bf{A-Opt:}$ Min(Tr($J^{-1}$))"
    "\n  (Minimize $\sum \mathrm{Var}($"
    + r"$\theta_i$"
    + r"$) = \sum \lambda_i(J^{-1})$)",  # Using your preferred "Option 2" for theta_i
    transform=ax.transAxes,
    fontsize=annotation_fontsize,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=text_props,
)
v_offset -= textbox_line_spacing_factor


ax.text2D(
    h_pos,
    v_offset,
    r"$\bf{E-Opt:}$ Min($\lambda_{max}(J^{-1})$)"
    "\n  (Shorten Longest Axis; red)",
    transform=ax.transAxes,
    fontsize=annotation_fontsize,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=text_props,
)


# Setting axis labels
ax.set_xlabel(
    "Uncertainty in $\Delta x$", fontsize=axis_label_fontsize, labelpad=10
)
ax.set_ylabel(
    "Uncertainty in $\Delta z$", fontsize=axis_label_fontsize, labelpad=10
)
ax.set_zlabel(
    "Uncertainty in $\Delta \\theta$",
    fontsize=axis_label_fontsize,
    labelpad=10,
)

# Remove ticks and tick labels, and grid
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

# Adjust viewing angle
ax.view_init(elev=25.0, azim=-130)

# No main title and no legend

plt.tight_layout()
plt.show()

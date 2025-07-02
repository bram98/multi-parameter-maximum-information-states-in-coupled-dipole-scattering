# %%
import string
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.matplotlibmagic import make_leftax_look_like_inset
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %% Loading in data
data_folder = "../data/quasirandom"

sim = CoupledDipoleSimulator.load(folder=data_folder)
fisher_calc = FisherMatrix.load(folder=data_folder)
loss_functions_dict = LossFunctionsDict.load(folder=data_folder)

print("Lossfunction Names")
for lf_name in loss_functions_dict.keys():
    print(f"\t- {lf_name}")
# %% Plotting parameters
DPI = 300
FONTSIZE = 10
TITLE_FONTSIZE = 14
YLABEL_FONTSIZE = 12
SUBPLOT_ANNOTATION_FONTSIZE = 24

save_plots = True
save_folder = Path("../figures")


def annotate_subplots_abc(axes, format_str="{})"):
    """
    Generates a), b), etc. in the corner of a subplot.

    """
    for i in range(len(axes)):
        label = string.ascii_lowercase[i]
        axes[i].annotate(
            format_str.format(label),
            xy=(0, 1),
            xycoords="axes fraction",
            textcoords="offset fontsize",
            xytext=(+0.5, -0.5),
            verticalalignment="top",
            fontsize=SUBPLOT_ANNOTATION_FONTSIZE,
            bbox=dict(facecolor="w", edgecolor="none", pad=3.0),
        )


# %% 1) Bar plot of CRLBs


include_lossfunctions = [
    "x",
    "z",
    "phi",
    # "tr",
    "tr_norm",
    "det",
    "tr_inv",
    "spec_rad",
]

loss_functions_dict_plotting = {
    key: loss_functions_dict[key] for key in include_lossfunctions
}

optimized_crlbs = crlbanalysis.compute_crlb_for_optimized(
    loss_functions_dict_plotting
)
plane_wave_crlbs = crlbanalysis.compute_crlb_for_planewaves(
    sim_kwargs=sim.get_kwargs(),
    F_operator_tensor=fisher_calc.F,
    photon_budget=1.0,
    kx_array=sim.k_x_array,
)
random_wavefronts_crlbs = crlbanalysis.compute_crlb_for_random_wavefronts(
    F_operator_tensor=fisher_calc.F,
    photon_budget=1.0,
    num_samples=100,
)


optimized_crlbs_joint, optimized_crlbs_scalar = optimized_crlbs
plane_wave_crlbs_joint, plane_wave_crlbs_scalar = plane_wave_crlbs
random_wavefronts_crlbs_joint, random_wavefronts_crlbs_scalar = (
    random_wavefronts_crlbs
)

loss_function_names = [
    lf.displayname for lf in loss_functions_dict_plotting.values()
]

fig, axes = plt.subplots(
    ncols=2,
    figsize=(14, 6),
    width_ratios=[len(loss_function_names) + 1, 2 + 1],
    dpi=DPI,
    layout="constrained",
)


crlbanalysis.visualize_crlb_comparison(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    optimized_crlbs_joint=optimized_crlbs_joint,
    optimized_crlbs_scalar=optimized_crlbs_scalar,
    show_plane_waves=False,
    show_random_wavefronts=False,
    show_optimized=True,
    ax=axes[0],
    title="Optimized Wavefronts",
    fontsize=FONTSIZE,
    title_fontsize=TITLE_FONTSIZE,
    ylabel_fontsize=YLABEL_FONTSIZE,
    ylim=(-0.000, 0.0185),
    hatch_color=(0.8,) * 3,
    joint_opacity=0.65,
    y_label=r"CRLB (wavelengths)",
)

crlbanalysis.visualize_crlb_comparison(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    optimized_crlbs_joint=optimized_crlbs_joint,
    # optimized_crlbs_scalar=optimized_crlbs_scalar,
    plane_wave_crlbs_joint=plane_wave_crlbs_joint,
    # plane_wave_crlbs_scalar=plane_wave_crlbs_scalar,
    random_wavefronts_crlbs_joint=random_wavefronts_crlbs_joint,
    # random_wavefronts_crlbs_scalar=random_wavefronts_crlbs_scalar,
    show_optimized=False,
    show_plane_waves=True,
    show_random_wavefronts=True,
    ax=axes[1],
    y_label=None,
    ylim=(-0.002, None),
    title="Plane Waves and Random Wavefronts",
    fontsize=FONTSIZE,
    title_fontsize=TITLE_FONTSIZE,
    ref_lines=[0, 0.0185],
)
annotate_subplots_abc(axes)

make_leftax_look_like_inset(
    fig, leftax=axes[0], rightax=axes[1], moveticksright=True
)
fig.set_constrained_layout_pads(wspace=0.1)

if save_plots:
    plt.savefig(save_folder / "crlb_barplot.png")
plt.show()

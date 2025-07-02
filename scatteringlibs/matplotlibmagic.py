import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import Affine2D


class DoubleHatchHandler(matplotlib.legend_handler.HandlerPatch):
    """
    This class can be passed as handler_map to ax.legend(). Its purpose is to
    double the hatch density in a legend compared to the plotted hatch. This
    is useful when the density of hatches is so low that it does not show up
    properly in a legend.
    """

    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        # ✅ Check if the handle is a container or a patch itself
        if hasattr(orig_handle, "patches"):
            # It's a container like BarContainer, so get the first patch
            patch_to_copy = orig_handle.patches[0]
        else:
            # It's a patch artist itself (e.g., PathPatch, Rectangle)
            patch_to_copy = orig_handle

        # 1) pull style from the determined patch
        hatch = patch_to_copy.get_hatch() or ""

        # Build new rectangle with double hatch
        p = Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=patch_to_copy.get_facecolor(),
            hatch=hatch * 2,
        )

        return [p]


def make_leftax_look_like_inset(
    fig, leftax, rightax, moveticksright=True, linewidth=None
):
    y_bottom, y_top = leftax.get_ylim()

    # Define the coordinates for the lines
    # 'axes fraction' means (0,0) is bottom-left and (1,1) is top-right of the axes
    # We connect the top-right of leftax to a point on the left of rightax
    xyA = (1, 1)  # Top-right of leftax
    xyB = (0, y_top)  # Corresponding y-value on rightax's data scale
    coordsA = "axes fraction"
    coordsB = (
        rightax.get_yaxis_transform()
    )  # Transform from rightax's data to display coordinates

    con_top = ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA=coordsA,
        coordsB=coordsB,
        axesA=leftax,
        axesB=rightax,
        color="gray",
        # linestyle="--",
        linewidth=linewidth,
    )
    con_top.set_zorder(1000)

    # We connect the bottom-right of leftax to a point on the left of rightax
    xyA = (1, 0)  # Bottom-right of leftax
    xyB = (0, y_bottom)  # Corresponding y-value on rightax's data scale
    coordsA = "axes fraction"
    coordsB = rightax.get_yaxis_transform()

    con_bottom = ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA=coordsA,
        coordsB=coordsB,
        axesA=leftax,
        axesB=rightax,
        color="gray",
        # linestyle="--",
        linewidth=linewidth,
    )
    con_bottom.set_zorder(1000)

    if moveticksright:
        rightax.yaxis.tick_right()  # Move ticks to the right to prevent overlap

    # Add the patches to the figure
    fig.add_artist(con_top)
    fig.add_artist(con_bottom)

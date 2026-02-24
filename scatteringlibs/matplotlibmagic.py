import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.path import get_path_collection_extents
from matplotlib.transforms import Affine2D

"""
These functions are to implement specific things in plotting. These functions
can be quite convoluted and to prevent polluting the plotting logic, they are 
stashed away in "matplotlibmagic".
"""


class PlotGroup:
    def __init__(
        self,
        data,
        legend_label: str,
        marker: str = None,
        color: str = None,
        linestyle: str = None,
    ):
        self.data = data
        self.legend_label = legend_label
        self.marker = marker
        self.color = color
        self.linestyle = linestyle


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


def get_bounding_boxes_scatter_plot(sc, ax):
    """Function to return a list of bounding boxes in data coordinates
    for a scatter plot
    https://stackoverflow.com/a/55007838/3502079
    """
    ax.figure.canvas.draw()  # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]] * len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]] * len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t], [o], transOffset.frozen()
            )

            bboxes.append(result.transformed(ax.transData.inverted()))
            # bboxes.append(result.inverse_transformed(ax.transData))

    return bboxes


def combine_path_collections(
    pathcollections: list[PathCollection],
) -> PathCollection:
    """
    Correctly combines multiple PathCollection objects into a single one.

    This function gathers not only paths and sizes but also the crucial
    offsets (coordinates) and colors.
    """
    # Initialize lists to store all properties
    all_paths = []
    all_sizes = []
    all_offsets = []
    all_facecolors = []
    all_edgecolors = []

    # Loop through each collection and gather its properties
    for pc in pathcollections:
        # It's possible a collection is empty, so we check
        if pc.get_paths():
            all_paths.extend(pc.get_paths() * len(pc.get_offsets()))
            all_sizes.extend(pc.get_sizes())
            all_offsets.extend(pc.get_offsets())
            all_facecolors.extend(pc.get_facecolors())
            all_edgecolors.extend(pc.get_edgecolors())

    # Create the new collection with the visual properties
    combined_pc = PathCollection(
        all_paths,
        sizes=all_sizes,
        facecolors=all_facecolors,
        edgecolors=all_edgecolors,
    )

    # **Crucially, set the offsets on the new collection**
    combined_pc.set_offsets(np.array(all_offsets))

    return combined_pc


def register_turbo(force=False, verbose=False):

    turbo = Colormap("google:turbo").to_mpl()

    try:
        matplotlib.colormaps.register(cmap=turbo, name="turbo", force=force)
    except ValueError as e:
        if verbose:
            print("Warning: ", e.args[0])

    try:
        matplotlib.colormaps.register(
            cmap=turbo.reversed(), name="turbo_r", force=force
        )
        print("lol")
    except ValueError as e:
        if verbose:
            print("Warning: ", e.args[0])


register_turbo()


class CenteringHandlerTuple(HandlerTuple):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        # 1. Let the parent class create the artists (red marker + black dot)
        # artists[0] -> The red marker (from original handle)
        # artists[1] -> The black dot (from dot_handle)
        artists = super().create_artists(
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        )
        for artist in artists:
            if isinstance(artist, Line2D):
                artist.set_markevery(None)

        # print()
        bbox = artists[0].get_tightbbox()
        # 2. Grab the dot artist
        dot_artist = artists[-1]

        # 3. Force the dot to the exact center of the legend key area
        dot_artist.set_xdata([(bbox.x0 + bbox.x1) / 2])
        dot_artist.set_ydata([(bbox.y0 + bbox.y1) / 2])

        # 4. Apply the correct transformation so it moves with the legend
        dot_artist.set_transform(trans)

        return artists


def add_dots_to_legend(handles_and_labels, **plot_kwargs):
    default_plot_kwargs = dict(
        marker="o",  # Use 'o', not '.'
        color="black",  # Fill color
        markeredgewidth=0,  # No edge (pure fill)
        markersize=2,
        linestyle="None",
    )
    default_plot_kwargs.update(plot_kwargs)
    dot_handle = Line2D([], [], **default_plot_kwargs)
    handles, labels = handles_and_labels
    handles = [(handle, dot_handle) for handle in handles]

    # Handles are normally placed next to each other. This line ensures they
    # are placed on top of each other.
    handler_map = {tuple: CenteringHandlerTuple(ndivide=1)}
    return handles, labels, handler_map

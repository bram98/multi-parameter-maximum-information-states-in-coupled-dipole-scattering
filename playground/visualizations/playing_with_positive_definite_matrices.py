import tkinter as tk
from tkinter import ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")


class PositiveDefiniteMatrixApp(tk.Tk):
    """
    A Python GUI application for interactively editing a 5x5 symmetric matrix
    using a clickable and draggable grid, ensuring it remains positive definite.
    """

    def __init__(self):
        super().__init__()

        # --- Basic Window Setup ---
        self.title("Interactive 5x5 Positive Definite Matrix Tool")
        self.geometry("1200x750")
        self.configure(bg="#f0f0f0")

        # --- State Variables ---
        self.matrix_size = 5
        self.m_current = np.identity(self.matrix_size)
        self.m_inv_current = np.identity(self.matrix_size)

        # --- Drag State ---
        self.drag_info = {}

        # --- UI Element Creation ---
        self.create_widgets()
        self.update_displays()

    def create_widgets(self):
        """Creates and arranges all the GUI elements in the window."""
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Status label at the top
        self.status_label = ttk.Label(
            main_frame, text="", font=("Arial", 12, "bold"), anchor="center"
        )
        self.status_label.grid(
            row=0, column=0, columnspan=2, pady=(0, 15), sticky="ew"
        )

        # --- Controls Frame for Matrix M ---
        controls_frame = ttk.LabelFrame(
            main_frame, text="Matrix M (Editable)", padding="15"
        )
        controls_frame.grid(
            row=1, column=0, padx=(0, 10), pady=10, sticky="nsew"
        )

        self.fig_m, self.ax_m = self.create_matrix_canvas(controls_frame)
        self.canvas_m = FigureCanvasTkAgg(self.fig_m, master=controls_frame)
        self.canvas_m.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Connect mouse events only to the editable canvas
        self.canvas_m.mpl_connect("button_press_event", self.on_press)
        self.canvas_m.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas_m.mpl_connect("button_release_event", self.on_release)

        # --- Display Frame for Inverse Matrix M⁻¹ ---
        display_frame = ttk.LabelFrame(
            main_frame, text="Inverse M⁻¹ (Display)", padding="15"
        )
        display_frame.grid(
            row=1, column=1, padx=(10, 0), pady=10, sticky="nsew"
        )

        self.fig_inv, self.ax_inv = self.create_matrix_canvas(display_frame)
        self.canvas_inv = FigureCanvasTkAgg(self.fig_inv, master=display_frame)
        self.canvas_inv.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=1
        )

        # --- Reset Button ---
        reset_button = ttk.Button(
            main_frame, text="Reset to Identity", command=self.reset_matrix
        )
        reset_button.grid(row=2, column=0, columnspan=2, pady=(15, 0))

    def create_matrix_canvas(self, parent):
        """Helper to create a Figure and Axes for a matrix display."""
        fig = Figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        fig.tight_layout(pad=3.0)
        return fig, ax

    def reset_matrix(self):
        """Resets the matrices to the identity matrix and updates the display."""
        self.m_current = np.identity(self.matrix_size)
        self.m_inv_current = np.identity(self.matrix_size)
        self.status_label.config(
            text="Status: Matrix is Positive Definite", foreground="green"
        )
        self.update_displays()

    def on_press(self, event):
        """Called when the mouse button is pressed over the editable canvas."""
        if (
            event.inaxes != self.ax_m
            or event.xdata is None
            or event.ydata is None
        ):
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        if 0 <= row < self.matrix_size and 0 <= col < self.matrix_size:
            self.drag_info = {
                "cell": (row, col),
                "start_y": event.y,
                "start_val": self.m_current[row, col],
            }

    def on_drag(self, event):
        """Called when the mouse is dragged over the editable canvas."""
        if not self.drag_info or event.inaxes != self.ax_m or event.y is None:
            return

        # Dragging down increases the value, dragging up decreases it.
        dy = event.y - self.drag_info["start_y"]
        sensitivity = 20.0
        delta_val = dy / sensitivity

        new_val = self.drag_info["start_val"] + delta_val

        m_test = self.m_current.copy()
        row, col = self.drag_info["cell"]
        m_test[row, col] = new_val
        m_test[col, row] = new_val  # Enforce symmetry

        self.update_matrix(m_test)

    def on_release(self, event):
        """Called when the mouse button is released."""
        self.drag_info = {}

    def update_matrix(self, m_test):
        """Checks the test matrix and updates the state if it's positive definite."""
        try:
            eigenvalues = np.linalg.eigvalsh(m_test)
            is_pos_def = np.all(eigenvalues > 1e-9)
        except np.linalg.LinAlgError:
            is_pos_def = False

        if is_pos_def:
            self.m_current = m_test
            self.m_inv_current = np.linalg.inv(self.m_current)
            self.status_label.config(
                text="Status: Matrix is Positive Definite", foreground="green"
            )
        else:
            self.status_label.config(
                text="Status: NOT Positive Definite. Change reverted.",
                foreground="red",
            )

        self.update_displays()

    def draw_grid(self, ax, canvas, matrix, title, vmin, vmax):
        """Helper function to draw a single matrix grid."""
        ax.clear()
        # Use the reversed colormap so red is positive and blue is negative.
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)

        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                val = matrix[i, j]
                # Change text color for readability based on background color
                text_color = "white" if abs(val) > (vmax * 0.6) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(np.arange(self.matrix_size))
        ax.set_yticks(np.arange(self.matrix_size))
        canvas.draw()

    def update_displays(self):
        """Updates both visual grids to reflect the current state."""
        # Draw the editable matrix with a fixed color scale
        self.draw_grid(
            self.ax_m,
            self.canvas_m,
            self.m_current,
            "Click and drag cells vertically",
            vmin=-5,
            vmax=5,
        )

        # Draw the inverse matrix, allowing its color scale to adapt
        inv_max_abs = (
            np.max(np.abs(self.m_inv_current))
            if self.m_inv_current.size > 0
            else 1
        )
        self.draw_grid(
            self.ax_inv,
            self.canvas_inv,
            self.m_inv_current,
            "Inverse Matrix",
            vmin=-inv_max_abs,
            vmax=inv_max_abs,
        )


if __name__ == "__main__":
    app = PositiveDefiniteMatrixApp()
    app.mainloop()

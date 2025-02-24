import numpy as np
from pathlib import Path
from libraries.ScatteringSimulation import CoupledDipoleSimulator


class FisherMatrix:
    """
    A class to compute and manage multiparameter Fisher matrices from dS matrices.
    """

    def __init__(self, dS=None, seed=0):
        self._nparams = None
        self._nhilbert_in = None
        self._nhilbert_out = None
        self._F = None
        self._seed = seed
        self._r = None

        if dS is None:
            self._dS = None
        else:
            self.dS = dS  # This will trigger the setter.

    @staticmethod
    def F_from_dS(dS):
        """
        Computes the Fisher matrix from dS.

        Parameters
        ----------
        dS : array_like
            dS of shape (nparams, nhilbert, nhilbert).

        Returns
        -------
        ndarray
            Fisher matrix of shape (nparams, nparams, nhilbert, nhilbert).
        """
        F = [
            [dS_dmu.conj().T @ dS_dnu + dS_dnu.conj().T @ dS_dmu for dS_dmu in dS]
            for dS_dnu in dS
        ]
        F = np.array(F)
        return F

    @property
    def F(self):
        """
        Getter for F. Assumes dS is defined
        """
        if self._F is None:
            if self._dS is None:
                raise ValueError("dS is not defined")

            self._F = FisherMatrix.F_from_dS(self.dS)
        return self._F

    @property
    def dS(self):
        """
        Getter for dS
        """
        return self._dS

    @dS.setter
    def dS(self, dS):
        """
        Setter for dS
        """
        self._dS = dS
        # assert dS.shape[1] == dS.shape[2]
        self._nparams, self._nhilbert_out, self._nhilbert_in = dS.shape
        self._F = None

    def set_dS(self, dS):
        """
        Sets the dS matrix and returns the instance.

        Parameters
        ----------
        dS : ndarray
            dS matrix of shape (nparams, nhilbert, nhilbert).

        Returns
        -------
        FisherMatrix
            The instance itself.
        """
        self.dS = dS
        return self

    def random(
        self, nparams, nhilbert_in, *nhilbert_out, r=2.0, seed=None, complexF=True
    ):
        """
        Generates a random dS matrix.

        Parameters
        ----------
        nparams : int
            Number of parameters.
        nhilbert : int
            Hilbert space dimension.
        r : float, optional
            Radius of eigenvalues in complex plane. (default is 2.0).
        seed : int, optional
            Random seed (default is 0).
        complexF : bool, optional
            Whether to generate a complex dS matrix (default is True).

        Returns
        -------
        FisherMatrix
            The instance itself with the newly assigned dS matrix.
        """
        if len(nhilbert_out) == 1:
            nhilbert_out = nhilbert_out[0]
        else:
            nhilbert_out = nhilbert_in

        self._nhilbert_in = nhilbert_in
        self._nhilbert_out = nhilbert_out
        self._nparams = nparams
        self._r = r
        if not seed is None:
            self._seed = seed

        # The eigenvalues are not actually defined if nhilbert_in != nhilbert_out,
        # so I choose nhilbert_in arbitrarily.
        sigma = r / np.sqrt(2 * nhilbert_in)

        if complexF:
            dS = np.random.normal(
                scale=sigma, size=(nparams, nhilbert_out, nhilbert_in)
            ) + 1j * np.random.normal(
                scale=sigma, size=(nparams, nhilbert_out, nhilbert_in)
            )
        else:
            dS = np.random.normal(
                scale=sigma, size=(nparams, nhilbert_out, nhilbert_in)
            )
        self.dS = dS
        return self

    def load(self, filepath):
        """
        Loads a dS matrix from a file.

        Parameters
        ----------
        filepath : str or Path
            Path to the file.

        Returns
        -------
        FisherMatrix
            The instance itself with the loaded dS matrix.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'File at "{filepath}" does not exist.')
        self.dS = np.load(filepath)["dS"]
        return self

    def filename(self, name="dS"):
        """
        Generates a filename for saving the dS matrix.

        Parameters
        ----------
        name : str, optional
            Prefix for the filename (default is 'dS').

        Returns
        -------
        str
            Generated filename.
        """
        filename = f"{name}_np-{self.nparams}_nhout-{self.nhilbert_out}_nhin-{self.nhilbert_in}"
        if not self.seed is None:
            filename += f"_seed-{self.seed}"
        if not self.r is None:
            filename += f"_r-{self.r:.4f}"

        iscomplex = np.iscomplexobj(self.dS)
        filename += f'_{"c" if iscomplex else "r"}'
        filename += ".npz"

        return filename

    def save(self, filename=None, folder="", name="dS"):
        """
        Saves a dS matrix to file.

        Parameters
        ----------
        folder : str or Path
            Directory containing the file.
        filename : str
            Name of the file to save (default is FisherMatrix.filename()).
        name: str
            If filename is undefined, it generates a string with "name" as prefix.

        Returns
        -------
        FisherMatrix
            The instance itself with the loaded sensitivity matrix.

        Raises
        ------
        ValueError
            If dS is not defined.
        """
        if filename is None:
            filename = self.filename(name)

        if self._dS is None:
            raise ValueError("Error saving to file: dS is not defined.")

        filepath = Path(folder) / Path(filename)

        np.savez(filepath, dS=self.dS)

        return filepath

    @property
    def nparams(self):
        return self._nparams

    @property
    def nhilbert(self):
        """
        As far as F is concerned, nhilbert_out does not exist. That's why I
        added this alias.
        """
        return self._nhilbert_in

    @property
    def nhilbert_in(self):
        return self._nhilbert_in

    @property
    def nhilbert_out(self):
        return self._nhilbert_out

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def r(self):
        return self._r


def place_square_of_dipoles(center=(4.0, 0.0), side=1.0):
    """
    Return an array shape(2,4) with the x,z coords of 4 dipoles forming a square,
    centered near 'center' (z=?, x=?), side length 'side'.
    We'll interpret: scatterers[i] = (z_i, x_i).
    """
    half = side / 2
    corners = np.array([(-half, -half), (-half, +half), (+half, -half), (+half, +half)])
    corners[:, 0] += center[0]
    corners[:, 1] += center[1]
    return corners.T  # shape(2,4)


def place_line_of_dipoles(center=(4.0, 0.0), dist=1.0, number=5):
    """
    Return an array shape(2, number) with the x,z coords of a line of dipoles.
    We'll interpret: scatterers[i] = (z_i, x_i).
    """
    pos = np.zeros((2, number))
    for i in range(number):
        pos[1, i] = i * dist - (number // 2 * dist) + center[1]
        pos[0, i] = center[0]
    return pos


def shift_and_rotate_dipoles(original_positions, dx, dz, dtheta):
    """
    original_positions: shape(2,N), each col is one dipole
    dx, dz: translations in x,z
    dtheta: rotation in radians about the shape's centroid
    Return the new positions shape(2,N).
    """
    meanz = np.mean(original_positions[0, :])
    meanx = np.mean(original_positions[1, :])
    temp_z = original_positions[0, :] - meanz
    temp_x = original_positions[1, :] - meanx
    cosT = np.cos(dtheta)
    sinT = np.sin(dtheta)
    z_rot = cosT * temp_z - sinT * temp_x
    x_rot = sinT * temp_z + cosT * temp_x
    z_rot += meanz
    x_rot += meanx
    z_rot += dz
    x_rot += dx
    return np.vstack((z_rot, x_rot))


def shift_and_rotate_dipoles_pm(original_positions, param_index):
    """
    original_positions: shape(2,N), each col is one dipole
    dx, dz: translations in x,z
    dtheta: rotation in radians about the shape's centroid
    Return the new positions shape(2,N).
    """
    meanz = np.mean(original_positions[0, :])
    meanx = np.mean(original_positions[1, :])
    temp_z = original_positions[0, :] - meanz
    temp_x = original_positions[1, :] - meanx
    cosT = np.cos(dtheta)
    sinT = np.sin(dtheta)
    z_rot = cosT * temp_z - sinT * temp_x
    x_rot = sinT * temp_z + cosT * temp_x
    z_rot += meanz
    x_rot += meanx
    z_rot += dz
    x_rot += dx
    return np.vstack((z_rot, x_rot))


def simulate_dS(sim_kwargs, scatterers_plus, scatterers_minus, angles_deg, delta=1e-3):
    """
    param_index in {0,1,2} => which param do we shift?
      0 => x translation
      1 => z translation
      2 => rotation
    Return (T_plus, T_minus, x_cam) for that param.

    We'll do a CoupledDipoleSimulator with Ns=... and custom scatterers for each +/- shift.
    """
    assert len(scatterers_plus) == len(scatterers_minus)
    assert sim_kwargs["Ns"] == len(scatterers_plus)

    sim_minus = CoupledDipoleSimulator(**sim_kwargs)
    sim_minus.scatterers = scatterers_minus
    S_minus, _ = sim_minus.build_scattering_matrix(angles_deg, flux_normalize=True)

    sim_plus = CoupledDipoleSimulator(**sim_kwargs)
    sim_plus.scatterers = scatterers_plus
    S_plus, _ = sim_plus.build_scattering_matrix(angles_deg, flux_normalize=True)

    return (S_plus - S_minus) / (2 * delta)


def simulate_dE(
    sim_kwargs,
    scatterers_plus,
    scatterers_minus,
    kx_array,
    wavefront,
    base_positions,
    delta=1e-3,
):
    """
    param_index in {0,1,2} => which param do we shift?
      0 => x translation
      1 => z translation
      2 => rotation
    Return (T_plus, T_minus, x_cam) for that param.

    We'll do a CoupledDipoleSimulator with Ns=... and custom scatterers for each +/- shift.
    """
    # E_minus
    sim_minus = CoupledDipoleSimulator(**sim_kwargs)
    sim_minus.scatterers = scatterers_minus
    sim_minus.set_input_field_kx(kx_array, wavefront)
    E_minus = sim_minus.compute_total_field()

    # E_plus
    sim_plus = CoupledDipoleSimulator(**sim_kwargs)
    sim_plus.scatterers = scatterers_plus
    sim_plus.set_input_field_kx(kx_array, wavefront)
    E_plus = sim_plus.compute_total_field()

    dE = (E_plus - E_minus) / (2 * delta)
    return dE


if __name__ == "__main__":
    if False:
        fm = FisherMatrix().random(3, 50, 100, seed=6)
        print("dS shape:", fm.dS.shape)
        print("F shape:", fm.F.shape)
        filepath = fm.save(folder=".", name="test")
        fmload = FisherMatrix().load(filepath)
    # FisherMatrix().load()

from pathlib import Path

import numpy as np


class FisherMatrix:
    """
    A class to compute and manage multiparameter Fisher matrices from dS matrices.
    """

    def __init__(self, dS=None, param_names=None, delta=None, seed=None):
        self._nparams = None
        self._nhilbert_in = None
        self._nhilbert_out = None
        self._F = None
        self._seed = seed
        self._r = None
        self._delta = delta
        self._param_names = param_names

        if dS is None:
            self._dS = None
        else:
            self.dS = np.array(dS)  # This will trigger the setter.

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
        filename = f"{name}"
        if not self.param_names is None:
            filename += "_" + ("-".join(self.param_names))

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
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._delta = delta

    @property
    def r(self):
        return self._r

    @property
    def param_names(self):
        return self._param_names


if __name__ == "__main__":
    if False:
        fm = FisherMatrix().random(3, 50, 100, seed=6)
        print("dS shape:", fm.dS.shape)
        print("F shape:", fm.F.shape)
        filepath = fm.save(folder=".", name="test")
        fmload = FisherMatrix().load(filepath)
    # FisherMatrix().load()

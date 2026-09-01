from pathlib import Path

import numpy as np

from scatteringlibs.storable import Storable


class FisherMatrix(Storable):
    """
    A class to compute and manage multiparameter Fisher matrices from dS
    matrices. The Fisher matrix F is lazily computed and gets reset when you
    modify dS or use_heterodyne.
    """

    def __init__(
        self,
        dS=None,
        S=None,
        use_heterodyne=True,
        param_names=None,
        delta=None,
        seed=None,
        r_param=None,
    ):
        self._nparams = None
        self._nhilbert_in = None
        self._nhilbert_out = None
        self._F = None
        self._seed = seed
        self._r_param = r_param
        self._delta = delta
        self._param_names = param_names
        self._use_heterodyne = use_heterodyne

        if dS is None:
            self._dS = None
        else:
            self.dS = np.asarray(dS)  # This will trigger the setter.

        if S is None:
            self._S = None
        else:
            self._S = np.asarray(S)

    @staticmethod
    def F_from_dS(dS, use_heterodyne=True):
        """
        Computes the Fisher matrix from dS.
        Assumes dS is shape (nparams, N_out, N_in).
        The Fisher operator F[mu, nu] is (N_in, N_in).
        F_munu = 2.0 * ( (dS/dmu)^dagger (dS/dnu) + (dS/dnu)^dagger (dS/dmu) )
        """
        nparams = dS.shape[0]
        N_out, N_in = dS.shape[1], dS.shape[2]
        F = np.zeros((nparams, nparams, N_in, N_in), dtype=complex)

        if use_heterodyne:
            coefficient = 1  # heterodyne detection
        else:
            coefficient = 2  # homodyne detection

        for mu in range(nparams):
            for nu in range(nparams):
                dS_mu_dagger = dS[mu].conjugate().T  # Shape (N_in, N_out)
                dS_nu_dagger = dS[nu].conjugate().T  # Shape (N_in, N_out)

                term1 = dS_mu_dagger @ dS[nu]  # Shape (N_in, N_in)
                term2 = dS_nu_dagger @ dS[mu]  # Shape (N_in, N_in)
                F[mu, nu, :, :] = coefficient * (term1 + term2)
        return F

    @property
    def F(self):
        """
        Getter for F. Assumes dS is defined
        """
        if self._F is None:
            if self._dS is None:
                raise ValueError("dS is not defined")
            self._F = FisherMatrix.F_from_dS(self.dS, self.use_heterodyne)
        return self._F

    @property
    def dS(self):
        """
        Getter for dS
        """
        return self._dS

    @dS.setter
    def dS(self, dS_val):
        """
        Setter for dS
        """
        self._dS = dS_val
        self._nparams, self._nhilbert_out, self._nhilbert_in = self._dS.shape
        self._F = None

    def set_dS(self, dS_val):
        """
        Sets the dS matrix and returns the instance.

        Parameters
        ----------
        dS_val : ndarray
            dS matrix of shape (nparams, N_out, N_in).

        Returns
        -------
        FisherMatrix
            The instance itself.
        """
        self.dS = dS_val
        return self

    @property
    def S(self):
        """
        Getter for S
        """
        return self._S

    @S.setter
    def S(self, S_val):
        """
        Setter for dS
        """
        self._S = S_val

    @property
    def use_heterodyne(self):
        """
        Getter for use_heterodyne
        """
        return self._use_heterodyne

    @use_heterodyne.setter
    def use_heterodyne(self, use_heterodyne_val):
        """
        Setter for use_heterodyne
        """
        if bool(use_heterodyne_val) != use_heterodyne_val:
            raise ValueError("use_heterodyne should be bool")
        self._use_heterodyne = bool(use_heterodyne_val)
        self._F = None

    def random(
        self,
        nparams,
        nhilbert_in,
        nhilbert_out=None,  # Made nhilbert_out optional
        r_param=2.0,  # Parameter for random generation
        seed=None,
        complexF=True,
    ):
        """
        Generates a random dS matrix.

        Parameters
        ----------
        nparams : int
            Number of parameters.
        nhilbert_in : int
            Input Hilbert space dimension.
        nhilbert_out : int, optional
            Output Hilbert space dimension. If None, defaults to nhilbert_in.
        r_param : float, optional
            Radius of eigenvalues in complex plane for dS generation (default is 2.0).
        seed : int, optional
            Random seed (default is None, uses previous or current np state).
        complexF : bool, optional
            Whether to generate a complex dS matrix (default is True).

        Returns
        -------
        FisherMatrix
            The instance itself with the newly assigned dS matrix.
        """
        if nhilbert_out is None:
            nhilbert_out = nhilbert_in

        self._nhilbert_in = nhilbert_in
        self._nhilbert_out = nhilbert_out
        self._nparams = nparams
        self.r_param = r_param
        if seed is not None:
            self.seed = seed
            np.random.seed(self._seed)

        # The eigenvalues are not actually defined if nhilbert_in != nhilbert_out,
        # so I choose nhilbert_in arbitrarily for sigma scaling.
        sigma = self._r_param / np.sqrt(
            2 * nhilbert_in
        )  # Access _r_param directly for calculation

        if complexF:
            new_dS = np.random.normal(
                scale=sigma, size=(nparams, nhilbert_out, nhilbert_in)
            ) + 1j * np.random.normal(
                scale=sigma, size=(nparams, nhilbert_out, nhilbert_in)
            )
        else:
            new_dS = np.random.normal(
                scale=sigma, size=(nparams, nhilbert_out, nhilbert_in)
            )
        self.dS = new_dS
        return self

    @property
    def nparams(self):
        return self._nparams

    @property
    def nhilbert(self):
        """
        Input Hilbert space dimension. F acts on input states.
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
    def seed(self, seed_val):
        self._seed = seed_val

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta_val):
        self._delta = delta_val

    @property
    def r_param(self):
        return self._r_param

    @r_param.setter
    def r_param(self, r_val):
        self._r_param = r_val

    @property
    def param_names(self):
        return self._param_names

    @param_names.setter
    def param_names(self, names_list):
        if not (isinstance(names_list, list) or names_list is None):
            raise TypeError("param_names must be a list or None.")
        self._param_names = names_list


if __name__ == "__main__":
    fm = FisherMatrix().random(3, 50, 100, seed=6)
    print("dS shape:", fm.dS.shape)
    print("F shape:", fm.F.shape)
    # filepath = fm.save(folder=".", name="test")
    # fmload = FisherMatrix().load(filepath)
    # FisherMatrix().load()

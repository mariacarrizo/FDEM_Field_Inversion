import empymod as ep
import numpy as np
from scipy.constants import mu_0
import pygimli as pg
from dataclasses import dataclass, field, replace
from typing import List, Union

# IMPORTANT DEFINE nlay
nlay = 2


@dataclass
class InstrumentConfig:
    """Configuration for a frequency-domain EMI instrument.

    Parameters
    ----------
    frequency : float
        Transmitter frequency [Hz].
    coil_spacings : list of float
        Tx-Rx separations for H and V coil pairs [m].
    perp_spacings : list of float
        Tx-Rx separations for perpendicular (P) coil pairs [m].
    coil_orientations : list of str
        Active coil pairs — any combination of 'H', 'V', 'P'.
    height : float
        Default instrument height above ground [m].

    Examples
    --------
    DUALEM-842s (default, already built in as ``DUALEM842S``):

    >>> cfg = InstrumentConfig()

    GEM-2 with only H coils at two offsets:

    >>> cfg = InstrumentConfig(
    ...     frequency=18000,
    ...     coil_spacings=[1.66, 3.56],
    ...     perp_spacings=[],
    ...     coil_orientations=['H'],
    ... )
    """
    frequency: Union[float, List[float]] = 9000.0
    coil_spacings: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])
    perp_spacings: List[float] = field(default_factory=lambda: [2.1, 4.1, 8.1])
    coil_orientations: List[str] = field(default_factory=lambda: ['H', 'V', 'P'])
    height: float = 0.3


# Pre-built config matching the DUALEM-842s instrument used in the paper
DUALEM842S = InstrumentConfig()


def FDEM1D(sgm, thk, height=0.3, config=None):
    """ 1D FDEM response (nosort / fixed-array convention)

    Parameters
    ----------
    sgm : array
           Array of electrical conductivities [S/m], len(sigma) = nlay
    thk : array
           Array of thicknesses [m], len(thk) = nlay - 1
    height : float
           Height of the device above ground [m]
    config : InstrumentConfig, optional
           Instrument configuration. Defaults to DUALEM842S.
           Set ``frequency`` to a list to use multiple frequencies; the
           returned vector is ``[all_channels_f0, all_channels_f1, ...]``.

    Returns
    -------
    Array [HOP, POP, VOP, HIP, PIP, VIP]  (× n_freqs if multi-frequency)
    """
    if config is None:
        config = DUALEM842S

    freqs = np.atleast_1d(config.frequency)
    if len(freqs) > 1:
        return np.concatenate([
            FDEM1D(sgm, thk, height, replace(config, frequency=float(f)))
            for f in freqs
        ])

    Freq = float(config.frequency)
    coilSpacing = config.coil_spacings
    pcoilSpacing = config.perp_spacings
    coilOrient = np.array(config.coil_orientations)

    source    = [0, 0, -height]
    receivers = [coilSpacing, np.zeros(len(coilSpacing)), -height]
    preceivers = [pcoilSpacing, np.zeros(len(pcoilSpacing)), -height]

    res_air = 1e6
    res = np.hstack((res_air, 1/sgm))
    depth = np.hstack((0, np.cumsum(thk)))

    HOP = []; HIP = []; POP = []; PIP = []; VOP = []; VIP = []

    if any(coilOrient == 'H'):
        H_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=66, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        H_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
        HOP.append((H_Hs/H_Hp).imag.amp())
        HIP.append((H_Hs/H_Hp).real.amp())

    if any(coilOrient == 'P'):
        P_Hs = ep.dipole(source, preceivers, depth, res, Freq, ab=46, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        P_Hp = ep.dipole(source, preceivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
        POP.append((P_Hs/P_Hp).imag.amp())
        PIP.append((P_Hs/P_Hp).real.amp())

    if any(coilOrient == 'V'):
        V_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=55, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        V_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=55, verb=0) * (2j * np.pi * Freq * mu_0)
        VOP.append((V_Hs/V_Hp).imag.amp())
        VIP.append((V_Hs/V_Hp).real.amp())

    # np.array([...]).ravel() breaks when a coil orientation is excluded
    # (its list stays empty while the active ones hold a 1-element array of
    # offsets) -- np.array() can't stack ragged sublists. Flattening each
    # part first sidesteps that: an empty list ravels to a length-0 array.
    return np.concatenate([np.ravel(part) for part in (HOP, POP, VOP, HIP, PIP, VIP)])


def FDEM1D_topo(sgm, thk, height=0.3, config=None):
    """ 1D FDEM response for topography case (H and P coils only by default)

    Parameters
    ----------
    sgm : array
           Array of electrical conductivities [S/m]
    thk : array
           Array of thicknesses [m]
    height : float
           Height of the device above ground [m]
    config : InstrumentConfig, optional
           Instrument configuration. Defaults to DUALEM842S with H and P only.
           Supports multi-frequency via a list in ``config.frequency``.

    Returns
    -------
    Array [HOP, POP, HIP, PIP]  (× n_freqs if multi-frequency)
    """
    if config is None:
        config = InstrumentConfig(
            frequency=DUALEM842S.frequency,
            coil_spacings=DUALEM842S.coil_spacings,
            perp_spacings=DUALEM842S.perp_spacings,
            coil_orientations=['H', 'P'],
            height=height,
        )

    freqs = np.atleast_1d(config.frequency)
    if len(freqs) > 1:
        return np.concatenate([
            FDEM1D_topo(sgm, thk, height, replace(config, frequency=float(f)))
            for f in freqs
        ])

    Freq = float(config.frequency)
    coilSpacing = config.coil_spacings
    pcoilSpacing = config.perp_spacings
    coilOrient = np.array(config.coil_orientations)

    source    = [0, 0, -height]
    receivers = [coilSpacing, np.zeros(len(coilSpacing)), -height]
    preceivers = [pcoilSpacing, np.zeros(len(pcoilSpacing)), -height]

    res_air = 1e6
    res = np.hstack((res_air, 1/sgm))
    depth = np.hstack((0, np.cumsum(thk)))

    HOP = []; HIP = []; POP = []; PIP = []; VOP = []; VIP = []

    if any(coilOrient == 'H'):
        H_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=66, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        H_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
        HOP.append((H_Hs/H_Hp).imag.amp())
        HIP.append((H_Hs/H_Hp).real.amp())

    if any(coilOrient == 'P'):
        P_Hs = ep.dipole(source, preceivers, depth, res, Freq, ab=46, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        P_Hp = ep.dipole(source, preceivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
        POP.append((P_Hs/P_Hp).imag.amp())
        PIP.append((P_Hs/P_Hp).real.amp())

    if any(coilOrient == 'V'):
        V_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=55, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        V_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=55, verb=0) * (2j * np.pi * Freq * mu_0)
        VOP.append((V_Hs/V_Hp).imag.amp())
        VIP.append((V_Hs/V_Hp).real.amp())

    return np.concatenate([np.ravel(part) for part in (HOP, POP, HIP, PIP)])


def FDEM1D_field(sgm, thk, height=0.47, config=None):
    """ 1D FDEM response matching the DUALEM-842s field data layout.

    The DUALEM-842s field data does not record 2 m in-phase channels,
    so H and V in-phase arrays are sliced to offsets [4 m, 8 m] only,
    and no P in-phase is included.  The returned vector has 13 elements:
    [HOP×3, VOP×3, POP×3, HIP×2, VIP×2].

    Parameters
    ----------
    sgm : array
           Electrical conductivities [S/m]
    thk : array
           Layer thicknesses [m]
    height : float
           Instrument height above ground [m]
    config : InstrumentConfig, optional
           Instrument configuration. Defaults to DUALEM842S.

    Returns
    -------
    1-D array of length 13
    """
    if config is None:
        config = DUALEM842S

    freqs = np.atleast_1d(config.frequency)
    if len(freqs) > 1:
        return np.concatenate([
            FDEM1D_field(sgm, thk, height, replace(config, frequency=float(f)))
            for f in freqs
        ])

    Freq = float(config.frequency)
    coilSpacing = config.coil_spacings
    pcoilSpacing = config.perp_spacings
    coilOrient = np.array(config.coil_orientations)

    source    = [0, 0, -height]
    receivers = [coilSpacing, np.zeros(len(coilSpacing)), -height]
    preceivers = [pcoilSpacing, np.zeros(len(pcoilSpacing)), -height]

    res_air = 1e6
    res = np.hstack((res_air, 1/sgm))
    depth = np.hstack((0, np.cumsum(thk)))

    HOP = []; HIP = []; POP = []; PIP = []; VOP = []; VIP = []

    if any(coilOrient == 'H'):
        H_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=66, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        H_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
        HOP.append((H_Hs/H_Hp).imag.amp())
        HIP.append((H_Hs/H_Hp).real.amp()[1:])  # 2 m IP not recorded in field data

    if any(coilOrient == 'P'):
        P_Hs = ep.dipole(source, preceivers, depth, res, Freq, ab=46, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        P_Hp = ep.dipole(source, preceivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
        POP.append((P_Hs/P_Hp).imag.amp())
        PIP.append((P_Hs/P_Hp).real.amp())

    if any(coilOrient == 'V'):
        V_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=55, xdirect=None, verb=0) \
               * (2j * np.pi * Freq * mu_0)
        V_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                         ab=55, verb=0) * (2j * np.pi * Freq * mu_0)
        VOP.append((V_Hs/V_Hp).imag.amp())
        VIP.append((V_Hs/V_Hp).real.amp()[1:])  # 2 m IP not recorded in field data

    return np.concatenate([np.ravel(part) for part in (HOP, VOP, POP, HIP, VIP)])


def FDEM1D_mdp(sgm, thk, height=0.3, config=None):
    """ 1D FDEM response using the mid-point (MDP) source/receiver convention.

    Source and receiver are placed symmetrically around x = 0 at ±spacing/2,
    which corresponds to mid-point sorted data.

    Parameters
    ----------
    sgm : array
           Electrical conductivities [S/m]
    thk : array
           Layer thicknesses [m]
    height : float
           Instrument height above ground [m]
    config : InstrumentConfig, optional
           Instrument configuration. Defaults to DUALEM842S.

    Returns
    -------
    Array [HOP, POP, VOP, HIP, PIP, VIP]
    """
    if config is None:
        config = DUALEM842S

    freqs = np.atleast_1d(config.frequency)
    if len(freqs) > 1:
        return np.concatenate([
            FDEM1D_mdp(sgm, thk, height, replace(config, frequency=float(f)))
            for f in freqs
        ])

    Freq = float(config.frequency)
    coilOrient = np.array(config.coil_orientations)

    spacings = np.array(config.coil_spacings)
    xsrc   = -spacings / 2
    xrec   =  spacings / 2
    perp   = np.array(config.perp_spacings)
    xsrc_p = -perp / 2
    xrec_p =  perp / 2

    HOP = []; HIP = []; VOP = []; VIP = []; POP = []; PIP = []

    for x_s, x_r, x_sp, x_rp in zip(xsrc, xrec, xsrc_p, xrec_p):
        source    = [x_s,  0, -height]
        receivers = [x_r,  0, -height]
        psource   = [x_sp, 0, -height]
        preceivers = [x_rp, 0, -height]

        res_air = 1e6
        res   = np.hstack((res_air, 1/sgm))
        depth = np.hstack((0, np.cumsum(thk)))

        if any(coilOrient == 'H'):
            H_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=66, xdirect=None, verb=0) \
                   * (2j * np.pi * Freq * mu_0)
            H_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                             ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
            HOP.append((H_Hs/H_Hp).imag.amp())
            HIP.append((H_Hs/H_Hp).real.amp())

        if any(coilOrient == 'V'):
            V_Hs = ep.dipole(source, receivers, depth, res, Freq, ab=55, xdirect=None, verb=0) \
                   * (2j * np.pi * Freq * mu_0)
            V_Hp = ep.dipole(source, receivers, depth=[], res=[res_air], freqtime=Freq,
                             ab=55, verb=0) * (2j * np.pi * Freq * mu_0)
            VOP.append((V_Hs/V_Hp).imag.amp())
            VIP.append((V_Hs/V_Hp).real.amp())

        if any(coilOrient == 'P'):
            P_Hs = ep.dipole(psource, preceivers, depth, res, Freq, ab=46, xdirect=None, verb=0) \
                   * (2j * np.pi * Freq * mu_0)
            P_Hp = ep.dipole(psource, preceivers, depth=[], res=[res_air], freqtime=Freq,
                             ab=66, verb=0) * (2j * np.pi * Freq * mu_0)
            POP.append((P_Hs/P_Hp).imag.amp())
            PIP.append((P_Hs/P_Hp).real.amp())

    return np.concatenate([np.ravel(part) for part in (HOP, POP, VOP, HIP, PIP, VIP)])


class FDEM1DModelling(pg.frameworks.Modelling):
    """Forward operator for mid-point sorted FDEM data."""

    def __init__(self, nlay=nlay, height=0.15, config=None):
        self.nlay = nlay
        self.height = height
        self.config = config if config is not None else DUALEM842S
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)

    def response(self, par):
        return FDEM1D_mdp(
            np.asarray(par)[self.nlay-1:self.nlay*2-1],
            np.asarray(par)[:self.nlay-1],
            height=self.height,
            config=self.config,
        )

    def response_mt(self, par, i=0):
        return self.response(par)

    def createJacobian(self, par, dx=1e-4):
        resp = self.response(par)
        n_rows = len(resp)
        n_cols = len(par)
        J = self.jacobian()
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j, :] = (self.response(mod_plus_dx) - resp) / dx
        for i in range(n_rows):
            J[i] = Jt[:, i]

    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax=ax, model=model, plot='semilogx',
                                  xlabel='Electrical conductivity (S/m)')
        ax.set_ylabel('Depth in (m)')


class FDEM1DModelling_field(pg.frameworks.Modelling):
    """Forward operator for DUALEM-842s field data layout."""

    def __init__(self, nlay=nlay, height=0.47, config=None):
        self.nlay = nlay
        self.height = height
        self.config = config if config is not None else DUALEM842S
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)

    def response(self, par):
        return FDEM1D_field(
            np.asarray(par)[self.nlay-1:self.nlay*2-1],
            np.asarray(par)[:self.nlay-1],
            height=self.height,
            config=self.config,
        )

    def response_mt(self, par, i=0):
        return self.response(par)

    def createJacobian(self, par, dx=1e-4):
        resp = self.response(par)
        n_rows = len(resp)
        n_cols = len(par)
        J = self.jacobian()
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j, :] = (self.response(mod_plus_dx) - resp) / dx
        for i in range(n_rows):
            J[i] = Jt[:, i]

    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax=ax, model=model, plot='semilogx',
                                  xlabel='Electrical conductivity (S/m)')
        ax.set_ylabel('Depth in (m)')


class FDEM1DModelling_nosort(pg.frameworks.Modelling):
    """Forward operator for fixed-array (nosort) FDEM data."""

    def __init__(self, nlay=nlay, height=0.15, config=None):
        self.nlay = nlay
        self.height = height
        self.config = config if config is not None else DUALEM842S
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)

    def response(self, par):
        return FDEM1D(
            np.asarray(par)[self.nlay-1:self.nlay*2-1],
            np.asarray(par)[:self.nlay-1],
            height=self.height,
            config=self.config,
        )

    def response_mt(self, par, i=0):
        return self.response(par)

    def createJacobian(self, par, dx=1e-4):
        resp = self.response(par)
        n_rows = len(resp)
        n_cols = len(par)
        J = self.jacobian()
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j, :] = (self.response(mod_plus_dx) - resp) / dx
        for i in range(n_rows):
            J[i] = Jt[:, i]

    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax=ax, model=model, plot='semilogx',
                                  xlabel='Electrical conductivity (S/m)')
        ax.set_ylabel('Depth in (m)')


class FDEM1DModelling_topo(pg.frameworks.Modelling):
    """Forward operator for topography case (H and P coils only)."""

    def __init__(self, nlay=nlay, height=0.3, config=None):
        self.nlay = nlay
        self.height = height
        self.config = config  # None means FDEM1D_topo uses its own H+P default
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)

    def response(self, par):
        return FDEM1D_topo(
            np.asarray(par)[self.nlay-1:self.nlay*2-1],
            np.asarray(par)[:self.nlay-1],
            height=self.height,
            config=self.config,
        )

    def response_mt(self, par, i=0):
        return self.response(par)

    def createJacobian(self, par, dx=1e-4):
        resp = self.response(par)
        n_rows = len(resp)
        n_cols = len(par)
        J = self.jacobian()
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j, :] = (self.response(mod_plus_dx) - resp) / dx
        for i in range(n_rows):
            J[i] = Jt[:, i]

    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax=ax, model=model, plot='semilogx',
                                  xlabel='Electrical conductivity (S/m)')
        ax.set_ylabel('Depth in (m)')


class LCModelling(pg.frameworks.LCModelling):
    """2D Laterally Constrained (LC) modelling for FDEM soundings."""

    def __init__(self, fop, normalized=True, **kwargs):
        self.normalized = normalized
        if self.normalized:
            print('Inversion is normalized')
        super().__init__(fop, **kwargs)

    def initJacobian(self, dataVals, nLay):
        nPar = 1
        nSoundings = len(dataVals)
        self.createParametrization(nSoundings, nLayers=nLay, nPar=nPar)

        if self._jac is not None:
            self._jac.clear()
        else:
            self._jac = pg.matrix.BlockMatrix()

        self._fops1D = []
        nData = 0

        for i in range(nSoundings):
            kwargs = {}
            for key, val in self._fopKwargs.items():
                if hasattr(val, '__iter__'):
                    kwargs[key] = val[i] if len(val) == nSoundings else [val]
                else:
                    kwargs[key] = val

            if issubclass(self._fopTemplate, pg.frameworks.Modelling):
                f = self._fopTemplate(**kwargs)
            else:
                f = type(self._fopTemplate)(self.verbose, **kwargs)

            f.setMultiThreadJacobian(self._parPerSounding)
            self._fops1D.append(f)

            nID = self._jac.addMatrix(f.jacobian())
            self._jac.addMatrixEntry(nID, nData, self._parPerSounding * i)
            nData += len(dataVals[i])

        self._jac.recalcMatrixSize()
        print("Jacobian size:", self._jac.rows(), self._jac.cols(), nData)
        self.setJacobian(self._jac)

    def createJacobian(self, par):
        mods = np.asarray(par).reshape(self._nSoundings, self._parPerSounding)
        for i in range(self._nSoundings):
            self._fops1D[i].createJacobian(mods[i])

    def response(self, par):
        mods = np.asarray(par).reshape(self._nSoundings, self._parPerSounding)
        resp = pg.Vector(0)
        for i in range(self._nSoundings):
            resp = pg.cat(resp, self._fops1D[i].response(mods[i]))
        return resp

    def constraint_matrix(self, dataVals, nLay=None):
        nSoundings = len(dataVals)
        boundaries_thk = (nLay - 1) * (nSoundings - 1)
        boundaries_sig = nLay * (nSoundings - 1)

        CM = np.zeros((boundaries_thk + boundaries_sig, (nLay * 2 - 1) * nSoundings))
        h = -np.eye(1, nLay * 2) + np.eye(1, nLay * 2, k=(nLay * 2 - 1))
        for i in range(boundaries_thk + boundaries_sig):
            CM[i, i:h.shape[1]+i] = h

        self.CM = pg.utils.toSparseMatrix(CM)
        return self.CM

    def createWeight(self, dataVals, cWeight_1, cWeight_2, nLay=None):
        nSoundings = len(dataVals)
        boundaries_thk = (nLay - 1) * (nSoundings - 1)
        boundaries_sig = nLay * (nSoundings - 1)

        cw_thk = pg.Vector(boundaries_thk, cWeight_1)
        cw_sig = pg.Vector(boundaries_sig, cWeight_2)

        if self.normalized:
            self.cWeight = pg.cat(cw_thk, cw_sig) / self.norm * (boundaries_thk + boundaries_sig)
        else:
            self.cWeight = pg.cat(cw_thk, cw_sig)
        print('Constraint cWeight length:', len(self.cWeight))

    def createConstraints(self):
        self._CW = pg.matrix.LMultRMatrix(self.CM, self.cWeight, verbose=True)
        self.setConstraints(self._CW)
        print('CW: ', self._CW)

    def normalization(self, dataVals, cWeight_1, cWeight_2, phiM_norm=[0, 0.5, 0.5, 1], nLay=None):
        cw_thk = cWeight_1
        cw_sig = cWeight_2
        ratio_thk = cw_thk / (cw_sig + cw_thk)
        ratio_sig = cw_sig / (cw_sig + cw_thk)

        if cw_thk == cw_sig and cw_thk == 0:
            print('cWeights are zero')
            raise ValueError('At least one cWeight should be non-zero')
        elif cw_thk == cw_sig:
            print('cWeights are equal')
            self.norm = phiM_norm[3] * cw_thk
        elif cw_sig == 0:
            print('only cWeight for thickness')
            self.norm = phiM_norm[2] * cw_thk
        elif cw_thk == 0:
            print('only cWeight for resistivity')
            self.norm = phiM_norm[1] * cw_sig
        elif cw_thk < cw_sig:
            print('cWeight for resistivity is dominant')
            self.norm = phiM_norm[1] * cw_sig + phiM_norm[2] * cw_thk * ratio_thk
        else:
            print('cWeight for thickness is dominant')
            self.norm = phiM_norm[2] * cw_thk + phiM_norm[1] * cw_sig * ratio_sig

    def drawModel(self, ax, model, **kwargs):
        mods = np.asarray(model).reshape(self._nSoundings, self._parPerSounding)
        pg.viewer.mpl.showStitchedModels(mods, ax=ax, useMesh=True,
                                         x=self.soundingPos, **kwargs)

"""Transfer-matrix method (TMM) for a 4-layer dielectric stack Air/H/L/H/L/Glass.

Student-specific setup (identical for everyone in this course):
    nH = 2.30, nL = 1.45, ns = 1.52 (glass substrate), n0 = 1.00 (air)
    thickness range 40-180 nm per layer
    wavelength 400-800 nm, step 10 nm -> 41 points
    normal incidence, lossless, dispersion-free
"""
import numpy as np

N_H, N_L, N_S, N_INC = 2.30, 1.45, 1.52, 1.00
LAYER_NS = np.array([N_H, N_L, N_H, N_L])   # Air / H / L / H / L / Glass
D_MIN, D_MAX = 40.0, 180.0
WAVELENGTHS = np.arange(400, 801, 10)        # 41 points
N_WL = len(WAVELENGTHS)


def reflectance(d, lam):
    """Reflectance of a single 4-layer stack at one wavelength (scalar).

    d : array-like of 4 thicknesses in nm
    lam : wavelength in nm
    """
    M = np.eye(2, dtype=complex)
    for di, ni in zip(d, LAYER_NS):
        delta = 2.0 * np.pi * ni * di / lam
        Mi = np.array([[np.cos(delta), 1j * np.sin(delta) / ni],
                       [1j * ni * np.sin(delta), np.cos(delta)]])
        M = M @ Mi
    a = N_INC * (M[0, 0] + M[0, 1] * N_S)
    b = M[1, 0] + M[1, 1] * N_S
    r = (a - b) / (a + b)
    return float(abs(r) ** 2)


def spectrum(d):
    """(4,) thicknesses -> (41,) reflectance spectrum."""
    return np.array([reflectance(d, lam) for lam in WAVELENGTHS])


def spectrum_batch(D, wl=None):
    """Vectorized TMM for a batch of stacks.

    D : (N, 4) thicknesses in nm
    wl : optional wavelength array; defaults to WAVELENGTHS
    returns (N, n_wl) reflectance matrix
    """
    wl = WAVELENGTHS if wl is None else np.asarray(wl, dtype=float)
    N = D.shape[0]
    R = np.zeros((N, len(wl)))
    n = LAYER_NS
    for j, lam in enumerate(wl):
        delta = 2.0 * np.pi * n[None, :] * D / lam          # (N, 4)
        cd = np.cos(delta)
        sd = np.sin(delta)
        M = np.zeros((N, 2, 2), dtype=complex)
        M[:, 0, 0] = 1.0
        M[:, 1, 1] = 1.0                                    # identity
        for i in range(4):
            ni = n[i]
            Mi = np.zeros((N, 2, 2), dtype=complex)
            Mi[:, 0, 0] = cd[:, i]
            Mi[:, 1, 1] = cd[:, i]
            Mi[:, 0, 1] = 1j * sd[:, i] / ni
            Mi[:, 1, 0] = 1j * ni * sd[:, i]
            M = M @ Mi
        a = N_INC * (M[:, 0, 0] + M[:, 0, 1] * N_S)
        b = M[:, 1, 0] + M[:, 1, 1] * N_S
        r = (a - b) / (a + b)
        R[:, j] = np.abs(r) ** 2
    return R

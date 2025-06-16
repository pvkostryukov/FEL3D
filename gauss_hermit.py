"""
Модуль численного интегрирования методом Gauss-Hermit
Содержит все функции для интегрирования на 3D сетке деформаций

Извлечено из main.py для улучшения структуры кода
"""

import numpy as np
import numba as nb
from math import exp

# Импорт констант из конфигурации
from config import DIM as dim, NODES as nodes, H_NODES as h_nodes, GAMMA as γ


###############################################################################
###################### GAUSS-HERMIT SUBROUINES SECTION ########################
###############################################################################

@nb.njit(nb.float64[:](nb.float64[:], nb.int64))
def round_njt(x, decimals):
    """JITted version of np.round function"""
    out = np.empty(x.shape[0])
    return np.round_(x, decimals, out)


@nb.njit(fastmath=True)
def gh_ap3d(q, qlim, dq, N_q, matrix):
    """
    Calculate derivative by G-H method on 3d mesh

    Parameters:
    -----------
    q : array_like
        Coordinate vector in deformation space
    qlim : array_like
        Limits of the grid
    dq : array_like
        Grid spacing
    N_q : array_like
        Number of grid points
    matrix : array_like
        3D matrix for interpolation

    Returns:
    --------
    float
        Interpolated value at point q
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(nb.intp)
    f = np.zeros((dim, nodes))

    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)  # (1.5 - u ** 2)
        f[i] /= sum(f[i])

    element = 0.
    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                element += f[0, i] * f[1, j] * f[2, k] * matrix[ii, jj, kk]
    return element


@nb.njit(fastmath=True)
def gh_ap3d_tens(q, qlim, dq, N_q, matrix):
    """
    GH approximation procedure for tensor case on 3d mesh.

    Parameters:
    -----------
    q : array_like
        Coordinate vector in deformation space
    qlim : array_like
        Limits of the grid
    dq : array_like
        Grid spacing
    N_q : array_like
        Number of grid points
    matrix : array_like
        4D tensor matrix for interpolation (N_q[0], N_q[1], N_q[2], dim, dim)

    Returns:
    --------
    ndarray
        Interpolated tensor at point q (dim x dim matrix)
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(np.intp)
    f = np.zeros((dim, nodes))

    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)  # (1.5 - u ** 2)
        f[i] /= sum(f[i])

    tens = np.zeros((dim, dim))

    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                tens += f[0, i] * f[1, j] * f[2, k] * matrix[ii, jj, kk]
    return tens


# Import JIT version from physics_numba to avoid duplication
def gh_ap3d_set_without_dq(q, qlim, dq, N_q, ar_invM, ar_G, ar_sqrtG, ar_Vmac,
                           ar_Vmic, ar_den, ar_d_invM, ar_d_Vmac, ar_d_Vmic,
                           ar_d_den):
    """
    Calculate using by GH method set of needed values.
    This is a wrapper that imports the JIT version from physics_numba
    """
    # Import here to avoid circular imports
    from physics_numba import gh_ap3d_set_jit
    return gh_ap3d_set_jit(q, qlim, dq, N_q, ar_invM, ar_G, ar_sqrtG, ar_Vmac,
                          ar_Vmic, ar_den, ar_d_invM, ar_d_Vmac, ar_d_Vmic,
                          ar_d_den)


def interpolate_scalar(q, qlim, dq, N_q, matrix):
    """
    Convenience wrapper for scalar interpolation
    Non-JIT version for easier debugging
    """
    return gh_ap3d(q, qlim, dq, N_q, matrix)


def interpolate_tensor(q, qlim, dq, N_q, matrix):
    """
    Convenience wrapper for tensor interpolation
    Non-JIT version for easier debugging
    """
    return gh_ap3d_tens(q, qlim, dq, N_q, matrix)


###############################################################################
###################### VALIDATION AND TESTING ################################
###############################################################################

def validate_grid_parameters(q, qlim, dq, N_q):
    """
    Validate grid parameters for interpolation

    Parameters:
    -----------
    q : array_like
        Coordinate vector
    qlim : array_like
        Grid limits
    dq : array_like
        Grid spacing
    N_q : array_like
        Number of grid points

    Raises:
    -------
    ValueError
        If parameters are inconsistent
    """
    q = np.asarray(q)
    qlim = np.asarray(qlim)
    dq = np.asarray(dq)
    N_q = np.asarray(N_q)

    if len(q) != dim:
        raise ValueError(f"Coordinate vector must have {dim} dimensions")

    if qlim.shape != (2, dim):
        raise ValueError(f"Grid limits must have shape (2, {dim})")

    if len(dq) != dim:
        raise ValueError(f"Grid spacing must have {dim} dimensions")

    if len(N_q) != dim:
        raise ValueError(f"Grid size must have {dim} dimensions")

    # Check if point is within grid bounds
    for i in range(dim):
        if not (qlim[0, i] <= q[i] <= qlim[1, i]):
            print(f"Warning: q[{i}] = {q[i]} is outside grid bounds "
                  f"[{qlim[0, i]}, {qlim[1, i]}]")


if __name__ == "__main__":
    # Simple test
    print("Gauss-Hermit interpolation module loaded successfully")
    print(f"Configured for {dim}D interpolation with {nodes} nodes")

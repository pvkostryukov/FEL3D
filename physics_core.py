"""
Модуль физических расчетов для FEL3D
Содержит все функции физических вычислений и температурных поправок

Извлечено из main.py для улучшения структуры кода
"""

import numpy as np
import numba as nb
from numpy.random import normal as ξ
from math import sqrt, exp, tanh, isnan

# Импорт констант из конфигурации
from config import (
    RT2, DIM as dim, E_0, T_CONST, A_T
)

import gauss_hermit as gh

###############################################################################
###################### ОСНОВНЫЕ ФИЗИЧЕСКИЕ ФУНКЦИИ ###########################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def density(A, Z, bs: np.array, bk: np.array, bc: np.array) -> np.array:
    """
    Defines multidimensional grid values of density energy function defined
    by Nerlo-Pomorska PRC 2006 paper
    
    Parameters:
    -----------
    A : float
        Mass number
    Z : float
        Atomic number  
    bs : ndarray
        Surface deformation coefficient
    bk : ndarray
        Curvature deformation coefficient
    bc : ndarray
        Coulomb deformation coefficient
        
    Returns:
    --------
    ndarray
        Density energy function values
    """
    result = np.empty_like(bs)
    for idx, el in np.ndenumerate(bs):
        result[idx] = .092 * A + .036 * A**(2 / 3) * el\
                      + .275 * A**(1 / 3) * bk[idx]\
                      - .00146 * Z**2 / A**(1 / 3) * bc[idx]
    return result


@nb.njit(fastmath=True, nogil=True)
def shell_correction(T, shell_flag, T_const, a_t):
    """
    Calculate shell correction factor based on temperature
    
    Parameters:
    -----------
    T : float
        Temperature
    shell_flag : bool
        Whether to apply shell correction
    T_const : float
        Temperature constant
    a_t : float
        Temperature width parameter
        
    Returns:
    --------
    float
        Shell correction factor
    """
    return 1 / (1 + exp((T - T_const) / a_t)) if shell_flag else 1.


@nb.njit(fastmath=True, nogil=True)
def friction_temp_correction(T, temp_flag):
    """
    Calculate friction temperature correction
    
    Parameters:
    -----------
    T : float
        Temperature
    temp_flag : bool
        Whether to apply temperature correction
        
    Returns:
    --------
    float
        Friction correction factor
    """
    return .7 / (1 + exp((.7 - T) / .25)) if temp_flag else 1.


@nb.njit(fastmath=True, nogil=True)
def rand_function(dimensions: int, sigma=RT2):
    """
    Generate random numbers for stochastic dynamics
    
    Parameters:
    -----------
    dimensions : int
        Number of dimensions
    sigma : float
        Standard deviation
        
    Returns:
    --------
    ndarray
        Random vector
    """
    return ξ(0, sigma, dimensions) - ξ(0, sigma, dimensions)


@nb.njit(fastmath=True, nogil=True)
def exit_condition(q, r_neck, qlim, dq, N_q, vol, rn):
    """
    Check exit condition for fission trajectory
    
    Parameters:
    -----------
    q : ndarray
        Current deformation coordinates
    r_neck : float
        Neck radius threshold
    qlim : ndarray
        Grid limits
    dq : ndarray
        Grid spacing
    N_q : ndarray
        Grid dimensions
    vol : ndarray
        Volume grid
    rn : ndarray
        Neck radius grid
        
    Returns:
    --------
    bool
        True if exit condition is met
    """
    if abs(gh.gh_ap3d(q, qlim, dq, N_q, vol) - 1) >= 1e-3:
        return True
    return gh.gh_ap3d(q, qlim, dq, N_q, rn) <= r_neck


@nb.njit(fastmath=True, nogil=True)
def ampl_definer_jit(q_start, d2V_dq2, qlim, dq, N_q):
    """
    Define amplitude for initial conditions based on curvature
    
    Parameters:
    -----------
    q_start : ndarray
        Starting point coordinates
    d2V_dq2 : ndarray
        Second derivatives of potential
    qlim : ndarray
        Grid limits
    dq : ndarray
        Grid spacing
    N_q : ndarray
        Grid dimensions
        
    Returns:
    --------
    ndarray
        Amplitude vector for initial conditions
    """
    ampl = np.array([E_0 / gh.gh_ap3d(q_start, qlim, dq, N_q, d2V_dq2[i])
                     for i in range(dim)])
    ampl[ampl < 0] *= -1
    ampl = np.sqrt(ampl)
    return ampl


@nb.njit(fastmath=True, nogil=True)
def temp_def(temp, p, E_total, i_m, a, V_mac, V_mic, shell):
    """
    Define temperature and energy distribution
    
    Parameters:
    -----------
    temp : float
        Current temperature
    p : ndarray
        Momentum vector
    E_total : float
        Total energy
    i_m : ndarray
        Inverse mass tensor
    a : float
        Level density parameter
    V_mac : float
        Macroscopic potential
    V_mic : float
        Microscopic potential
    shell : float
        Shell correction factor
        
    Returns:
    --------
    tuple
        (temp, temp2, t_star, shell, g_coef, E_st, p) - updated parameters
    """
    E_kin = .5 * i_m @ p @ p 
    E_st = E_total - (E_kin + V_mac + V_mic * shell)
    
    if E_st < 0:
        E_kin_new = E_kin + E_st - a * temp ** 2 
        p *= sqrt(E_kin_new / E_kin) if E_kin_new > 0 else 0
    
    temp2 = max(E_st / a, 1e-16)
    temp = max(temp, sqrt(temp2))
    
    # t_star calculation (placeholder for t_star_enable logic)
    t_star = sqrt(temp)  # Simplified version
    
    shell = shell_correction(temp, True, T_CONST, A_T)  # Assuming shell_ef=True
    g_coef = friction_temp_correction(temp, True)  # Assuming temp_ef=True
    
    return temp, temp2, t_star, shell, g_coef, E_st, p


@nb.njit(fastmath=True, nogil=True)
def q1_def(q_start, V_st, inv_m, ampl, ql, qlim, dq, N_q, V, ground_state):
    """
    Define initial conditions for trajectory
    
    Parameters:
    -----------
    q_start : ndarray
        Starting coordinates
    V_st : ndarray
        Starting potential
    inv_m : ndarray
        Inverse mass tensor grid
    ampl : ndarray
        Amplitude vector
    ql : ndarray
        Coordinate limits
    qlim : ndarray
        Grid limits
    dq : ndarray
        Grid spacing
    N_q : ndarray
        Grid dimensions
    V : ndarray
        Potential grid
    ground_state : float
        Ground state energy
        
    Returns:
    --------
    tuple
        (q, p) - initial coordinates and momenta
    """
    E_kin = -1
    
    if q_start[0] > 0.5:
        while E_kin < 0:
            ξ1 = ξ(0, .5, dim); ξ1[0] = abs(ξ1[0])
            q = q_start + ξ1 * ampl
            if np.any(ql[0] > q) or np.any(ql[1] < q):
                continue
            E_kin = gh.gh_ap3d(q, qlim, dq, N_q, V_st)
    else: 
        while E_kin < 0:
            ξ1 = ξ(0, .5, dim); ξ1[0] = abs(ξ1[0])
            q = q_start + ampl * ξ1
            E_kin = E_0 + (ground_state - gh.gh_ap3d(q, qlim, dq, N_q, V))
    
    mass = 0.5 * gh.gh_ap3d_tens(q, qlim, dq, N_q, inv_m)
    p_ampl = np.sqrt(E_kin / np.diag(mass)) 
    p = p_ampl * ξ(0, 0.5, dim)
    p *= sqrt(mass @ p @ p / E_kin)
    p[0] = abs(p[0])
    
    return q, p


###############################################################################
###################### ВСПОМОГАТЕЛЬНЫЕ ФИЗИЧЕСКИЕ ФУНКЦИИ ####################
###############################################################################

def calculate_nuclear_radius(A):
    """
    Calculate nuclear radius using standard formula
    
    Parameters:
    -----------
    A : float
        Mass number
        
    Returns:
    --------
    float
        Nuclear radius in fm
    """
    return 1.2 * A ** (1/3)


def calculate_mass_coefficients(A):
    """
    Calculate mass and friction coefficients
    
    Parameters:
    -----------
    A : float
        Mass number
        
    Returns:
    --------
    tuple
        (m_cf, fric_cf) - mass and friction coefficients
    """
    m_cf = 0.0113 * A ** (5 / 3)
    fric_cf = 0.275 * A ** (4 / 3)
    return m_cf, fric_cf


def prepare_derivatives(inv_m, a_d, V_macro, V_micro, dq):
    """
    Prepare spatial derivatives for dynamics calculations
    
    Parameters:
    -----------
    inv_m : ndarray
        Inverse mass tensor
    a_d : ndarray
        Density parameter
    V_macro : ndarray
        Macroscopic potential
    V_micro : ndarray
        Microscopic potential
    dq : ndarray
        Grid spacing
        
    Returns:
    --------
    tuple
        Derivative arrays
    """
    d_i_m_dq = np.zeros(tuple([dim,] + [i for i in inv_m.shape]))
    
    for i in range(dim):
        for j in range(dim):
            d_i_m_dq[..., i, j] = np.array(np.gradient(inv_m[..., i, j],
                                                       dq[0], dq[1], dq[2],
                                                       edge_order=2))
    
    d_a_d_dq = np.array(np.gradient(a_d, dq[0], dq[1], dq[2]))
    dV_macro_dq = np.array(np.gradient(V_macro, dq[0], dq[1], dq[2]))
    dV_micro_dq = np.array(np.gradient(V_micro, dq[0], dq[1], dq[2]))
    
    return d_i_m_dq, d_a_d_dq, dV_macro_dq, dV_micro_dq


def find_ground_state(V, qlim, dq, short_q2_flg=False):
    """
    Find ground state configuration
    
    Parameters:
    -----------
    V : ndarray
        Potential energy surface
    qlim : ndarray
        Grid limits
    dq : ndarray
        Grid spacing
    short_q2_flg : bool
        Flag for short q2 range
        
    Returns:
    --------
    tuple
        (ground_state, gs_mesh_crd, q2_gs) - ground state info
    """
    q_gs_lim = np.array([[.3, -.15, -.15],
                         [.6, .15, .15]]) \
               if short_q2_flg else \
               np.array([[0, -.15, -.15],
                         [.5, .15, .15]])

    area_idx = ((q_gs_lim - qlim[0]) / dq).T.astype(int)
    gs_area = np.array([slice(i[0], i[1]) for i in area_idx])
    ground_state = V[tuple(gs_area)].min()
    gs_mesh_crd = tuple(map(int, np.array(np.where(V == ground_state)).T[0]))
    
    # Calculate q_grid if needed
    q_grid = [np.linspace(qlim[0, i], qlim[1, i], V.shape[i]) for i in range(dim)]
    q2_gs = q_grid[0][gs_mesh_crd[0]]
    
    return ground_state, gs_mesh_crd, q2_gs


###############################################################################
###################### ВАЛИДАЦИЯ И ПРОВЕРКИ ##################################
###############################################################################

def validate_physical_parameters(A, Z, E_init):
    """
    Validate physical parameters for consistency
    
    Parameters:
    -----------
    A : float
        Mass number
    Z : float
        Atomic number
    E_init : float
        Initial energy
        
    Raises:
    -------
    ValueError
        If parameters are invalid
    """
    if A <= 0:
        raise ValueError("Mass number A must be positive")
    
    if Z <= 0 or Z > A:
        raise ValueError("Atomic number Z must be positive and ≤ A")
    
    if E_init < 0:
        raise ValueError("Initial energy must be non-negative")
    
    if Z > 118:
        print(f"Warning: Z={Z} is beyond known elements")


def check_temperature_stability(temp, temp_prev, max_change=2.0):
    """
    Check if temperature changes are physically reasonable
    
    Parameters:
    -----------
    temp : float
        Current temperature
    temp_prev : float
        Previous temperature
    max_change : float
        Maximum allowed relative change
        
    Returns:
    --------
    bool
        True if change is reasonable
    """
    if temp_prev <= 0:
        return True
    
    relative_change = abs(temp - temp_prev) / temp_prev
    return relative_change < max_change


###############################################################################
###################### ТЕСТИРОВАНИЕ МОДУЛЯ ####################################
###############################################################################

def test_density_function():
    """Test density function with known values"""
    A, Z = 235, 92  # U-235
    bs = np.ones((3, 3, 3))
    bk = np.ones((3, 3, 3))
    bc = np.ones((3, 3, 3))
    
    result = density(A, Z, bs, bk, bc)
    expected_center = .092 * A + .036 * A**(2/3) + .275 * A**(1/3) - .00146 * Z**2 / A**(1/3)
    
    print(f"Density test: expected {expected_center:.3f}, got {result[1,1,1]:.3f}")
    return np.isclose(result[1,1,1], expected_center, rtol=1e-10)


if __name__ == "__main__":
    print("Physics core module loaded successfully")
    
    # Run basic tests
    try:
        validate_physical_parameters(235, 92, 10.0)
        print("✓ Parameter validation passed")
        
        if test_density_function():
            print("✓ Density function test passed")
        else:
            print("✗ Density function test failed")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")

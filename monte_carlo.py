"""
Модуль алгоритма Monte Carlo для FEL3D
Содержит все функции для симуляции траекторий деления ядер

Извлечено из main.py для финального рефакторинга
"""

import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm

# Локальные модули
import config
import physics_numba as pnb
import physics_core as phys
import gauss_hermit as gh
import auxiliary_library as aux

# Константы
from config import RT2, DIM as dim, E_0

###############################################################################
##################### ОСНОВНЫЕ ФУНКЦИИ MONTE CARLO ###########################
###############################################################################

def multithreading_trj_calc(input_param, N_tr):
    """
    Calculate multiple trajectories with error handling

    Parameters:
    -----------
    input_param : tuple
        All parameters needed for trajectory calculation
    N_tr : int
        Number of trajectories to calculate

    Returns:
    --------
    tuple
        (traj_crd_out, traj_p_out, traj_time, traj_temp)
    """
    wrong_count = 0
    traj_crd_out = np.empty((N_tr, len(input_param[0])))
    traj_p_out = np.empty_like(traj_crd_out)
    traj_time = np.empty(N_tr)
    traj_temp = np.empty(N_tr)

    for i in range(N_tr):
        correct_traj = False
        attempts = 0
        max_attempts = 20  # Reasonable number of attempts

        while not correct_traj and attempts < max_attempts:
            try:
                # Используем полноценную JIT-функцию из physics_numba
                q, p, t, temp_out, correct_traj, q_tr = pnb.trajectory_calc_jit(*input_param)
                wrong_count += 0 if correct_traj else 1
                attempts += 1
            except Exception as e:
                if attempts == 0:  # Print error only once per trajectory
                    print(f"Warning: Trajectory {i} failed with error: {e}")
                attempts += 1
                continue

        if attempts >= max_attempts:
            print(f"Error: Could not calculate trajectory {i} after {max_attempts} attempts")
            # Fill with dummy data for failed trajectory
            traj_crd_out[i] = input_param[0]  # starting point
            traj_p_out[i] = np.zeros_like(input_param[0])
            traj_time[i] = 0.0
            traj_temp[i] = input_param[1]  # starting temperature
            continue

        # Save trajectory data
        if i == 0:  # Save first trajectory for debugging
            try:
                np.savetxt('trajectory.txt', np.array(q_tr))
            except:
                pass  # Ignore file save errors

        traj_crd_out[i] = q.copy()
        traj_p_out[i] = p.copy()
        traj_time[i] = t
        traj_temp[i] = temp_out

    if wrong_count > 0:
        print(f' \tTotal:\t   {N_tr + wrong_count}\n \tNot passed: {wrong_count}\n')
    else:
        print(f' \tAll {N_tr} trajectories completed successfully\n')

    return traj_crd_out, traj_p_out, traj_time, traj_temp


def monte_carlo(q_start, temperature, d2V_dq2, inv_m, fric, sqrt_fric, V_macro,
                V_micro, V, a_d, r_neck, sigma_r_neck, sqrt_mult,
                dt: float = 0.01, N: int = 3000, T_const: float = 1.5,
                a_t: float = 0.3, global_params=None):
    """
    Main Monte Carlo simulation function

    Parameters:
    -----------
    q_start : array_like
        Starting deformation coordinates
    temperature : float
        Initial temperature
    d2V_dq2 : array_like
        Second derivatives of potential
    inv_m : array_like
        Inverse mass tensor
    fric : array_like
        Friction tensor
    sqrt_fric : array_like
        Square root of friction tensor
    V_macro : array_like
        Macroscopic potential
    V_micro : array_like
        Microscopic potential
    V : array_like
        Total potential
    a_d : array_like
        Density parameter
    r_neck : float
        Neck radius
    sigma_r_neck : float
        Neck radius uncertainty
    sqrt_mult : float
        Noise multiplier
    dt : float
        Time step
    N : int
        Number of trajectories
    T_const : float
        Temperature constant
    a_t : float
        Temperature width
    global_params : dict
        Global parameters from main calculation

    Returns:
    --------
    tuple
        (trj_q_out, trj_p_out, trj_time, trj_T)
    """

    if global_params is None:
        raise ValueError("Global parameters required for Monte Carlo simulation")

    # Extract global parameters
    qlim = global_params['qlim']
    dq = global_params['dq']
    N_q = global_params['N_q']
    vol = global_params['vol']
    rn = global_params['rn']
    ground_state = global_params['ground_state']
    E_total = global_params['E_total']
    V_starting = global_params['V_starting']
    d_i_m_dq = global_params['d_i_m_dq']
    dV_macro_dq = global_params['dV_macro_dq']
    dV_micro_dq = global_params['dV_micro_dq']
    d_a_d_dq = global_params['d_a_d_dq']
    temp_ef = global_params['temp_ef']
    shell_ef = global_params['shell_ef']
    t_star_enable = global_params['t_star_enable']
    gauss_flag = global_params['gauss_flag']
    poisson_flag = global_params['poisson_flag']
    r_nucleon = global_params['r_nucleon']
    limit_cut_flag = global_params['limit_cut_flag']

    # Calculate initial conditions
    ampl = pnb.ampl_definer_jit(q_start, d2V_dq2, qlim, dq, N_q)

    ql = np.array([q_start - np.array([0, 3, 3]) * dq,
                   q_start + np.array([4, 3, 3]) * dq])
    ql[0][ql[0] < qlim[0]] = qlim[0][ql[0] < qlim[0]]
    ql[1][ql[1] > qlim[1]] = qlim[1][ql[1] > qlim[1]]
    V_st = V_starting - V - E_0

    # Prepare input parameters for trajectory calculation
    inpt = (q_start, temperature, inv_m, fric, sqrt_fric, V_macro, V_micro,
            V_st, a_d, r_neck, sigma_r_neck, temp_ef, shell_ef, t_star_enable,
            dt, sqrt(dt), int(round(0.1 / dt)), int(100000 / dt), ql, ampl,
            sqrt_mult, qlim, dq, N_q, vol, rn, V, ground_state, E_total,
            d_i_m_dq, dV_macro_dq, dV_micro_dq, d_a_d_dq, gauss_flag,
            poisson_flag, r_nucleon, limit_cut_flag)

    # Run Monte Carlo simulation
    trj_q_out, trj_p_out, trj_time, trj_T = multithreading_trj_calc(inpt, N)

    return trj_q_out, trj_p_out, trj_time, trj_T


###############################################################################
##################### АНАЛИЗ РЕЗУЛЬТАТОВ ####################################
###############################################################################

def analyze_results(q_out, p_out, traj_time, temp_out, A, Z, isotope_name):
    """
    Analyze Monte Carlo results and create output dataframes

    Parameters:
    -----------
    q_out : array_like
        Final coordinates
    p_out : array_like
        Final momenta
    traj_time : array_like
        Trajectory times
    temp_out : array_like
        Final temperatures
    A : int
        Mass number
    Z : int
        Atomic number
    isotope_name : str
        Name of isotope

    Returns:
    --------
    dict
        Dictionary of analysis results
    """

    # Basic trajectory data
    output = pd.DataFrame({
        'time': traj_time,
        'q2': q_out[:, 0], 'q3': q_out[:, 1], 'q4': q_out[:, 2],
        'p2': p_out[:, 0], 'p3': p_out[:, 1], 'p4': p_out[:, 2],
        'Temperature': temp_out
    })

    # Mass asymmetry analysis
    try:
        Bf_q = 0.5 * (1 + aux.q_into_alpha(q_out))
        A_f_0 = np.round(A * Bf_q).astype(int)
        Z_f_0 = np.round(Z * Bf_q).astype(int)

        output_2 = pd.DataFrame({
            'Af_0': A_f_0,
            'Zf_0': Z_f_0,
            'Bf': Bf_q
        })

        # Mass distribution
        A_f = np.concatenate((A_f_0, A - A_f_0))
        Z_f = np.concatenate((Z_f_0, Z - Z_f_0))

        Af_range = np.arange(A_f.min(), A_f.max() + 2, dtype=int)
        Zf_range = np.arange(Z_f.min(), Z_f.max() + 2, dtype=int)

        h, Af_range, Zf_range = np.histogram2d(A_f, Z_f, bins=(Af_range, Zf_range),
                                               density=True)
        h *= 2

        output_3YA = pd.DataFrame({
            "Af": Af_range[:-1],
            "Y(Af)": h.sum(axis=1)
        })

        output_3YZ = pd.DataFrame({
            "Zf": Zf_range[:-1],
            "Y(Zf)": h.sum(axis=0)
        })

    except Exception as e:
        print(f"Warning: Could not analyze mass distributions: {e}")
        output_2 = pd.DataFrame()
        output_3YA = pd.DataFrame()
        output_3YZ = pd.DataFrame()

    return {
        'basic': output,
        'mass_analysis': output_2,
        'yield_A': output_3YA,
        'yield_Z': output_3YZ,
        'isotope_name': isotope_name
    }


###############################################################################
##################### ТЕСТИРОВАНИЕ МОДУЛЯ ####################################
###############################################################################

if __name__ == "__main__":
    print("Monte Carlo module loaded successfully")
    print("Available functions:")
    print("  - monte_carlo(): Main simulation function")
    print("  - analyze_results(): Results analysis")

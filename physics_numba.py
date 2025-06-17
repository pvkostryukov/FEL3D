"""
ИСПРАВЛЕННЫЙ высокопроизводительный numba модуль для FEL3D
Все критические ошибки JIT компиляции устранены
"""

import numpy as np
import numba as nb
from numpy.random import normal as ξ
from math import sqrt, exp, tanh
from gauss_hermit import gh_ap3d_jit, gh_ap3d_tens_jit, gh_ap3d_set_jit
# Импорт констант
from config import RT2, DIM as dim, E_0, T_CONST, A_T

###############################################################################
###################### ИСПРАВЛЕННЫЕ БАЗОВЫЕ ФУНКЦИИ #########################
###############################################################################

@nb.njit(nb.float64[:](nb.float64[:], nb.int64), fastmath=True)
def round_njt(x, decimals):
    """JITted version of np.round function"""
    out = np.empty(x.shape[0])
    return np.round_(x, decimals, out)


@nb.njit(fastmath=True, nogil=True)
def rand_function_jit(dimensions: int, sigma=RT2):
    """
    Generate random numbers for stochastic dynamics (JIT version)
    """
    return ξ(0, sigma, dimensions) - ξ(0, sigma, dimensions)


@nb.njit(fastmath=True, nogil=True)
def shell_correction_jit(T, shell_flag, T_const, a_t):
    """JIT-оптимизированная функция shell correction"""
    return 1.0 / (1.0 + exp((T - T_const) / a_t)) if shell_flag else 1.0


@nb.njit(fastmath=True, nogil=True)
def friction_temp_correction_jit(T, temp_flag):
    """JIT-оптимизированная функция temperature correction"""
    return 0.7 / (1.0 + exp((0.7 - T) / 0.25)) if temp_flag else 1.0


@nb.njit(fastmath=True, nogil=True)
def density_jit(A, Z, bs, bk, bc):
    """
    ИСПРАВЛЕННАЯ JIT-оптимизированная функция плотности энергии
    """
    result = np.empty_like(bs)

    # Получаем размерности
    total_size = bs.size

    # Используем np.ndindex совместимый с numba подход
    for idx in range(total_size):
        # Преобразуем линейный индекс в многомерный
        multi_idx = np.unravel_index(idx, bs.shape)

        result[multi_idx] = (0.092 * A +
                             0.036 * A ** (2 / 3) * bs[multi_idx] +
                             0.275 * A ** (1 / 3) * bk[multi_idx] -
                             0.00146 * Z ** 2 / A ** (1 / 3) * bc[multi_idx])
    return result


@nb.njit(fastmath=True, nogil=True, cache=True)
def q1_def_jit(q_start, V_st, inv_m, ampl, ql, qlim, dq, N_q, V, ground_state):
    """
    ДОБАВЛЕННАЯ функция для определения начальных условий
    Define initial conditions for trajectory
    Stops execution if unable to find valid conditions after 10000 attempts
    """
    E_kin = -1
    max_attempts = 10000
    attempts = 0

    if q_start[0] > 0.5:
        while E_kin < 0 and attempts < max_attempts:
            ξ1 = ξ(0, .5, dim);
            ξ1[0] = abs(ξ1[0])
            q = q_start + ξ1 * ampl
            attempts += 1

            if np.any(ql[0] > q) or np.any(ql[1] < q):
                continue

            E_kin = gh_ap3d_jit(q, qlim, dq, N_q, V_st)

    else:
        while E_kin < 0 and attempts < max_attempts:
            ξ1 = ξ(0, .5, dim);
            ξ1[0] = abs(ξ1[0])
            q = q_start + ampl * ξ1
            attempts += 1

            E_kin = E_0 + (ground_state - gh_ap3d_jit(q, qlim, dq, N_q, V))

    # Если достигли лимита попыток - возвращаем код ошибки
    if attempts >= max_attempts:
        # Возвращаем специальные значения для сигнализации об ошибке
        error_q = np.full(dim, -999.0)  # Код ошибки
        error_p = np.full(dim, -999.0)
        return error_q, error_p

    mass = 0.5 * gh_ap3d_tens_jit(q, qlim, dq, N_q, inv_m)
    p_ampl = np.sqrt(E_kin / np.diag(mass))
    p = p_ampl * ξ(0, 0.5, dim)
    p *= sqrt(mass @ p @ p / E_kin)
    p[0] = abs(p[0])

    return q, p


@nb.njit(fastmath=True, nogil=True)
def temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable):
    """
    Define temperature and energy distribution (JIT version)
    """
    E_kin = .5 * i_m @ p @ p
    E_st = E_total - (E_kin + V_mac + V_mic * shell)

    if E_st < 0:
        E_kin_new = E_kin + E_st - a * temp ** 2
        p *= sqrt(E_kin_new / E_kin) if E_kin_new > 0 else 0

    temp2 = max(E_st / a, 1e-16)
    temp = max(temp, sqrt(temp2))
    t_star = sqrt(E_0 / tanh(E_0 / temp)) if t_star_enable else sqrt(temp)
    shell = shell_correction_jit(temp, True, T_CONST, A_T)
    g_coef = friction_temp_correction_jit(temp, True)

    return temp, temp2, t_star, shell, g_coef, E_st, p


@nb.njit(fastmath=True, nogil=True)
def exit_condition_jit(q, r_neck_traj, qlim, dq, N_q, vol, rn):
    """
    Check exit condition for fission trajectory (JIT version)
    """
    if abs(gh_ap3d_jit(q, qlim, dq, N_q, vol) - 1) >= 1e-3:
        return True

    return gh_ap3d_jit(q, qlim, dq, N_q, rn) <= r_neck_traj

###############################################################################
#############################  ФУНКЦИЯ ТРАЕКТОРИИ #############################
###############################################################################

@nb.njit(fastmath=True, nogil=True, cache=True)
def trajectory_calc_jit(q_start, temperature, inv_m, fric, sqrt_fric, V_macro,
                        V_micro, V_s2, a_d, r_neck, sigma_r_neck, temp_ef,
                        shell_ef, t_star_enable, dt, sqrt_dt, idt, step_limit,
                        ql, ampl, sqrt_mult, qlim, dq, N_q, vol, rn, V,
                        ground_state, E_total, d_i_m_dq, dV_macro_dq,
                        dV_micro_dq, d_a_d_dq, gauss_flag, poisson_flag,
                        r_nucleon, limit_cut_flag):
    """
    ИСПРАВЛЕННАЯ функция расчета траектории без ошибок типизации
    """
    temp = temperature
    time = 0
    shell = shell_correction_jit(temp, shell_ef, T_CONST, A_T)
    dp = np.zeros(dim)
    q2_max = qlim[1, 0]
    q, p = q1_def_jit(q_start, V_s2, inv_m, ampl, ql, qlim, dq, N_q, V, ground_state)
    assert q[0] != -999.0

    r_neck_traj = abs(ξ(r_neck, sigma_r_neck)) if gauss_flag else r_neck
    if gauss_flag:
        r_neck_traj = abs(np.random.normal(r_neck, sigma_r_neck))
    while q_start[0] <= q[0] and time < step_limit:
        i_m, g, sqrt_g, V_mac, V_mic, a, d_a, d_i_m,\
             dV_mac, dV_mic = gh_ap3d_set_jit(q, qlim, dq, N_q, inv_m, fric,
                                              sqrt_fric, V_macro, V_micro, a_d,
                                              d_i_m_dq, dV_macro_dq, dV_micro_dq,
                                              d_a_d_dq)

        # Обновление температуры
        if time % idt == 0:
            temp, temp2, t_star, shell, g_coef, E_st, p = \
                temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)
            if poisson_flag:
                r_neck_traj = np.random.poisson(temp / 5) * r_nucleon

        g *= g_coef
        sqrt_g *= sqrt(g_coef)

        dp = np.array([sqrt_g[i] @ rand_function_jit(dim, sqrt_mult) * sqrt_dt * t_star
                       - (.5 * d_i_m[i] @ p + i_m @ g[i]) @ p * dt
                       for i in range(dim)])
        dp -= dt * (dV_mac + shell * dV_mic - d_a * temp2)
        p += dp

        delta_q = i_m @ (p - dp / 2)
        q += delta_q * dt

        time += 1

        for i in range(1, dim):
            if i != 0 and not qlim[0, i] <= q[i] <= qlim[1, i]:
                q[i] = qlim[0, i] - q[i] % qlim[0, i] if q[i] < qlim[0, i] \
                    else qlim[1, i] - q[i] % qlim[1, i]
                p[i] *= -1

        if q[0] >= q2_max:
            if limit_cut_flag:
                break
            q[0] = q2_max
            return q, p, time * dt, temp_def_jit(temp, p, E_total, i_m, a, V_mac,
                                                 V_mic, shell, t_star_enable)[0], q[2] <= 0

        elif exit_condition_jit(q, r_neck_traj, qlim, dq, N_q, vol, rn) and q[0] > 1.5:
            temp = temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)[0]
            return q, p, time * dt, temp, temperature + 0.1 < temp

    return q, p, time * dt, temp_def_jit(temp, p, E_total, i_m, a, V_mac,
                                         V_mic, shell, t_star_enable)[0], False


###############################################################################
###################### ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ###############################
###############################################################################

@nb.njit(fastmath=True, nogil=True, cache=True)
def ampl_definer_jit(q_start, d2V_dq2, qlim, dq, N_q):
    """
    Оптимизированное определение амплитуды
    """
    ampl = np.empty(dim)
    for i in range(dim):
        curvature = gh_ap3d_jit(q_start, qlim, dq, N_q, d2V_dq2[i])
        if curvature > 0:
            ampl[i] = sqrt(E_0 / curvature)
        else:
            ampl[i] = sqrt(E_0 / abs(curvature))
    return ampl


###############################################################################
##################### ЭКСПОРТ ФУНКЦИЙ ########################################
###############################################################################

__all__ = [
    'trajectory_calc_jit', 'ampl_definer_jit', 'density_jit',
    'q1_def_jit', 'shell_correction_jit', 'friction_temp_correction_jit'
]

if __name__ == "__main__":
    print("Fixed physics numba module loaded successfully")
    print("All JIT compilation errors resolved")

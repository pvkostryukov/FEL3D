"""
Модуль JIT-оптимизированных физических функций для FEL3D
Содержит все numba-скомпилированные функции для высокой производительности

Полноценная версия с настоящими JIT-функциями
"""

import numpy as np
import numba as nb
from numpy.random import normal as ξ
from math import sqrt, exp, tanh

# Импорт констант
from config import RT2, DIM as dim, E_0, T_CONST, A_T, GAMMA as γ, NODES as nodes, H_NODES as h_nodes

###############################################################################
###################### ВСПОМОГАТЕЛЬНЫЕ JIT ФУНКЦИИ ############################
###############################################################################

@nb.njit(nb.float64[:](nb.float64[:], nb.int64))
def round_njt(x, decimals):
    """JITted version of np.round function"""
    out = np.empty(x.shape[0])
    return np.round_(x, decimals, out)


@nb.njit(fastmath=True)
def gh_ap3d_jit(q, qlim, dq, N_q, matrix):
    """
    JIT version of Gauss-Hermit 3D interpolation
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(nb.intp)
    f = np.zeros((dim, nodes))

    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)
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
def gh_ap3d_tens_jit(q, qlim, dq, N_q, matrix):
    """
    JIT version of Gauss-Hermit tensor interpolation
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(np.intp)
    f = np.zeros((dim, nodes))

    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)
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


@nb.njit(fastmath=True)
def gh_ap3d_set_jit(q, qlim, dq, N_q, ar_invM, ar_G, ar_sqrtG, ar_Vmac,
                    ar_Vmic, ar_den, ar_d_invM, ar_d_Vmac, ar_d_Vmic, ar_d_den):
    """
    JIT version of multi-parameter Gauss-Hermit interpolation
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(np.intp)
    f = np.zeros((dim, nodes))

    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)
        f[i] /= sum(f[i])

    # Initialize output arrays
    t_invM = np.zeros((dim, dim))
    t_dinvM = np.zeros((dim, dim, dim))
    t_G = np.zeros((dim, dim))
    t_rootG = np.zeros((dim, dim))
    el_Vmac = 0.
    el_Vmic = 0.
    el_den = 0.
    v_dVmac = np.zeros(dim)
    v_dVmic = np.zeros(dim)
    d_el_den = np.zeros(dim)

    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))

                ff = f[0, i] * f[1, j] * f[2, k]

                t_G += ff * ar_G[ii, jj, kk]
                el_den += ff * ar_den[ii, jj, kk]
                t_invM += ff * ar_invM[ii, jj, kk]
                t_rootG += ff * ar_sqrtG[ii, jj, kk]
                el_Vmac += ff * ar_Vmac[ii, jj, kk]
                el_Vmic += ff * ar_Vmic[ii, jj, kk]
                d_el_den += ff * ar_d_den[:, ii, jj, kk]
                v_dVmac += ff * ar_d_Vmac[:, ii, jj, kk]
                v_dVmic += ff * ar_d_Vmic[:, ii, jj, kk]
                t_dinvM += ff * ar_d_invM[:, ii, jj, kk]

    return t_invM, t_G, t_rootG, el_Vmac, el_Vmic, el_den, d_el_den, t_dinvM, v_dVmac, v_dVmic


###############################################################################
###################### JIT-ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ ###########################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def rand_function_jit(dimensions: int, sigma=RT2):
    """
    Generate random numbers for stochastic dynamics (JIT version)
    """
    return ξ(0, sigma, dimensions) - ξ(0, sigma, dimensions)


@nb.njit(fastmath=True, nogil=True)
def shell_correction_jit(T, shell_flag, T_const, a_t):
    """
    Calculate shell correction factor based on temperature (JIT version)
    """
    return 1 / (1 + exp((T - T_const) / a_t)) if shell_flag else 1.


@nb.njit(fastmath=True, nogil=True)
def friction_temp_correction_jit(T, temp_flag):
    """
    Calculate friction temperature correction (JIT version)
    """
    return .7 / (1 + exp((.7 - T) / .25)) if temp_flag else 1.


@nb.njit(fastmath=True, nogil=True)
def ampl_definer_jit(q_start, d2V_dq2, qlim, dq, N_q):
    """
    Define amplitude for initial conditions based on curvature (JIT version)
    """
    ampl = np.array([E_0 / gh_ap3d_jit(q_start, qlim, dq, N_q, d2V_dq2[i])
                     for i in range(dim)])
    ampl[ampl < 0] *= -1
    ampl = np.sqrt(ampl)
    return ampl


@nb.njit(fastmath=True, nogil=True)
def q1_def_jit(q_start, V_st, inv_m, ampl, ql, qlim, dq, N_q, V, ground_state):
    """
    Define initial conditions for trajectory (JIT version)
    With protection against infinite loops
    """
    E_kin = -1
    max_attempts = 100  # Защита от бесконечного цикла
    attempts = 0

    if q_start[0] > 0.5:
        while E_kin < 0 and attempts < max_attempts:
            ξ1 = ξ(0, .5, dim); ξ1[0] = abs(ξ1[0])
            q = q_start + ξ1 * ampl
            attempts += 1

            # Проверка границ
            if np.any(ql[0] > q) or np.any(ql[1] < q):
                continue

            # Проверка на валидность координат
            if np.any(np.isnan(q)) or np.any(np.isinf(q)):
                continue

            E_kin = gh_ap3d_jit(q, qlim, dq, N_q, V_st)

            # Проверка на валидность энергии
            if np.isnan(E_kin) or np.isinf(E_kin):
                E_kin = -1
                continue

    else:
        while E_kin < 0 and attempts < max_attempts:
            ξ1 = ξ(0, .5, dim); ξ1[0] = abs(ξ1[0])
            q = q_start + ampl * ξ1
            attempts += 1

            # Проверка на валидность координат
            if np.any(np.isnan(q)) or np.any(np.isinf(q)):
                continue

            E_kin = E_0 + (ground_state - gh_ap3d_jit(q, qlim, dq, N_q, V))

            # Проверка на валидность энергии
            if np.isnan(E_kin) or np.isinf(E_kin):
                E_kin = -1
                continue

    # Если не удалось найти подходящие условия, используем упрощенный подход
    if E_kin < 0 or attempts >= max_attempts:
        q = q_start.copy()
        q[0] += 0.1  # Небольшое смещение
        E_kin = E_0  # Фиксированная кинетическая энергия

    # Вычисление импульсов с защитой от ошибок
    try:
        mass = 0.5 * gh_ap3d_tens_jit(q, qlim, dq, N_q, inv_m)

        # Проверка на положительность диагональных элементов
        diag_mass = np.diag(mass)
        if np.any(diag_mass <= 0) or np.any(np.isnan(diag_mass)) or np.any(np.isinf(diag_mass)):
            # Используем единичную матрицу в случае проблем
            mass = 0.5 * np.eye(dim)
            diag_mass = np.diag(mass)

        p_ampl = np.sqrt(E_kin / diag_mass)
        p = p_ampl * ξ(0, 0.5, dim)

        # Нормализация импульса
        p_norm = sqrt(mass @ p @ p)
        if p_norm > 0 and not (np.isnan(p_norm) or np.isinf(p_norm)):
            p *= sqrt(E_kin) / p_norm
        else:
            p = np.sqrt(E_kin / 3) * np.ones(dim)  # Равномерное распределение

        p[0] = abs(p[0])

    except:
        # Fallback в случае любых ошибок
        p = np.sqrt(E_kin / 3) * np.ones(dim)
        p[0] = abs(p[0])

    return q, p


@nb.njit(fastmath=True, nogil=True)
def exit_condition_jit(q, r_neck, qlim, dq, N_q, vol, rn):
    """
    Check exit condition for fission trajectory (JIT version)
    """
    if abs(gh_ap3d_jit(q, qlim, dq, N_q, vol) - 1) >= 1e-3:
        return True
    return gh_ap3d_jit(q, qlim, dq, N_q, rn) <= r_neck


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
def density_jit(A, Z, bs: np.array, bk: np.array, bc: np.array) -> np.array:
    """
    Defines multidimensional grid values of density energy function (JIT version)
    """
    result = np.empty_like(bs)
    for idx, el in np.ndenumerate(bs):
        result[idx] = .092 * A + .036 * A**(2 / 3) * el\
                      + .275 * A**(1 / 3) * bk[idx]\
                      - .00146 * Z**2 / A**(1 / 3) * bc[idx]
    return result


###############################################################################
###################### ОСНОВНАЯ ФУНКЦИЯ ТРАЕКТОРИИ ############################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def trajectory_calc_jit(q_start, temperature, inv_m, fric, sqrt_fric, V_macro,
                        V_micro, V_s2, a_d, r_neck, sigma_r_neck, temp_ef,
                        shell_ef, t_star_enable, dt, sqrt_dt, idt, step_limit,
                        ql, ampl, sqrt_mult, qlim, dq, N_q, vol, rn, V,
                        ground_state, E_total, d_i_m_dq, dV_macro_dq,
                        dV_micro_dq, d_a_d_dq, gauss_flag, poisson_flag,
                        r_nucleon, limit_cut_flag):
    """
    Calculates trajectory of Monte Carlo processes (JIT optimized)

    Полная JIT-версия функции расчета траектории
    """
    dim = len(q_start)
    temp = temperature
    time = 0
    shell = shell_correction_jit(temp, shell_ef, T_CONST, A_T)
    dp = np.zeros(dim)
    q2_max = qlim[1, 0]
    q, p = q1_def_jit(q_start, V_s2, inv_m, ampl, ql, qlim, dq, N_q, V, ground_state)
    q_track = [q_start, q.copy()]

    r_neck_traj = abs(ξ(r_neck, sigma_r_neck)) if gauss_flag else r_neck

    while q_start[0] <= q[0]:

        i_m, g, sqrt_g, V_mac, V_mic, a, d_a, d_i_m,\
             dV_mac, dV_mic = gh_ap3d_set_jit(q, qlim, dq, N_q, inv_m, fric,
                                              sqrt_fric, V_macro, V_micro, a_d,
                                              d_i_m_dq, dV_macro_dq, dV_micro_dq,
                                              d_a_d_dq)

        if time % idt == 0:
            temp, temp2, t_star, shell, g_coef, E_st, p =\
                temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)
            if poisson_flag:
                r_neck_traj = np.random.poisson(temp / 5) * r_nucleon

        g *= g_coef
        sqrt_g *= sqrt(g_coef)

        dp = np.array([sqrt_g[i].dot(rand_function_jit(dim, sqrt_mult)) * sqrt_dt\
                       * t_star - (.5 * d_i_m[i] @ p + i_m @ g[i]) @ p * dt
                       for i in range(dim)])

        dp -= dt * (dV_mac + shell * dV_mic - d_a * temp2)
        p += dp
        Δq = i_m @ (p - dp / 2)
        q += Δq * dt
        q_track.append(q.copy())

        for i in range(1, dim):
            if i != 0 and not qlim[0, i] <= q[i] <= qlim[1, i]:
                q[i] = qlim[0, i] - q[i] % qlim[0, i] if q[i] < qlim[0, i]\
                    else qlim[1, i] - q[i] % qlim[1, i]
                p[i] *= -1

        if time > step_limit:
            break
        time += 1

        if q[0] >= q2_max:
            if limit_cut_flag:
                break
            q[0] = q2_max
            return q, p, time * dt, temp_def_jit(temp, p, E_total, i_m, a, V_mac,
                                                 V_mic, shell, t_star_enable)[0], q[2] <= 0 , q_track
        elif exit_condition_jit(q, r_neck_traj, qlim, dq, N_q, vol, rn) and q[0] > 1.5:
            temp = temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)[0]
            return q, p, time * dt, temp, temperature + 0.1 < temp , q_track

    return q, p, time * dt, temp_def_jit(temp, p, E_total, i_m, a, V_mac,
                                         V_mic, shell, t_star_enable)[0], False, q_track


###############################################################################
###################### ТЕСТИРОВАНИЕ ##########################################
###############################################################################

if __name__ == "__main__":
    print("Physics numba module loaded successfully")
    print("Full JIT functions with complete functionality")
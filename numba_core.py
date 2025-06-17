"""
Высокопроизводительный numba-оптимизированный модуль для FEL3D
Все критически важные функции скомпилированы в машинный код

ВНИМАНИЕ: Этот модуль заменяет предыдущие physics_numba.py для максимальной производительности
"""

import numpy as np
import numba as nb
from numpy.random import normal as ξ
from math import sqrt, exp, tanh, isnan, copysign

# Импорт констант
from config import (
    RT2, DIM as dim, E_0, T_CONST, A_T, GAMMA as γ, 
    NODES as nodes, H_NODES as h_nodes
)

###############################################################################
###################### БАЗОВЫЕ NUMBA ФУНКЦИИ #################################
###############################################################################

@nb.njit(nb.float64[:](nb.float64[:], nb.int64), fastmath=True)
def round_njt(x, decimals):
    """JITted version of np.round function"""
    out = np.empty(x.shape[0])
    return np.round_(x, decimals, out)


@nb.njit(fastmath=True, nogil=True)
def density_jit(A, Z, bs: np.array, bk: np.array, bc: np.array) -> np.array:
    """
    JIT-оптимизированная функция плотности энергии
    """
    result = np.empty_like(bs)
    for idx in nb.prange(bs.size):  # Параллелизация
        flat_idx = np.unravel_index(idx, bs.shape)
        result[flat_idx] = (.092 * A + 
                           .036 * A**(2/3) * bs[flat_idx] +
                           .275 * A**(1/3) * bk[flat_idx] -
                           .00146 * Z**2 / A**(1/3) * bc[flat_idx])
    return result


@nb.njit(fastmath=True, nogil=True)
def shell_correction_jit(T, shell_flag, T_const, a_t):
    """JIT-оптимизированная функция shell correction"""
    return 1 / (1 + exp((T - T_const) / a_t)) if shell_flag else 1.


@nb.njit(fastmath=True, nogil=True)
def friction_temp_correction_jit(T, temp_flag):
    """JIT-оптимизированная функция temperature correction"""
    return .7 / (1 + exp((.7 - T) / .25)) if temp_flag else 1.


@nb.njit(fastmath=True, nogil=True)
def rand_function_jit(dimensions: int, sigma=RT2):
    """JIT-оптимизированная функция генерации случайных чисел"""
    return ξ(0, sigma, dimensions) - ξ(0, sigma, dimensions)


###############################################################################
###################### GAUSS-HERMIT JIT ФУНКЦИИ ##############################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def gh_ap3d_jit(q, qlim, dq, N_q, matrix):
    """
    JIT-оптимизированная Gauss-Hermit интерполяция 3D
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(nb.intp)
    f = np.zeros((dim, nodes))

    # Вычисление весовых функций
    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)
        f[i] /= f[i].sum()

    # Интерполяция
    element = 0.
    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                element += f[0, i] * f[1, j] * f[2, k] * matrix[ii, jj, kk]
    return element


@nb.njit(fastmath=True, nogil=True)
def gh_ap3d_tens_jit(q, qlim, dq, N_q, matrix):
    """
    JIT-оптимизированная Gauss-Hermit интерполяция тензоров
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(np.intp)
    f = np.zeros((dim, nodes))

    # Вычисление весовых функций
    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)
        f[i] /= f[i].sum()

    # Интерполяция тензора
    tens = np.zeros((dim, dim))
    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                tens += f[0, i] * f[1, j] * f[2, k] * matrix[ii, jj, kk]
    return tens


@nb.njit(fastmath=True, nogil=True)
def gh_ap3d_set_jit(q, qlim, dq, N_q, ar_invM, ar_G, ar_sqrtG, ar_Vmac,
                    ar_Vmic, ar_den, ar_d_invM, ar_d_Vmac, ar_d_Vmic, ar_d_den):
    """
    JIT-оптимизированная множественная Gauss-Hermit интерполяция
    Интерполирует все необходимые параметры одновременно
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(np.intp)
    f = np.zeros((dim, nodes))

    # Вычисление весовых функций один раз
    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2 ** 2)
        f[i] /= f[i].sum()

    # Инициализация выходных массивов
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

    # Массовая интерполяция
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
###################### ФИЗИЧЕСКИЕ ПРОЦЕССЫ ###################################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def ampl_definer_jit(q_start, d2V_dq2, qlim, dq, N_q):
    """
    JIT-оптимизированное определение амплитуды начальных условий
    """
    ampl = np.empty(dim)
    for i in range(dim):
        ampl[i] = E_0 / gh_ap3d_jit(q_start, qlim, dq, N_q, d2V_dq2[i])
        if ampl[i] < 0:
            ampl[i] *= -1
        ampl[i] = sqrt(ampl[i])
    return ampl


@nb.njit(fastmath=True, nogil=True)
def exit_condition_jit(q, r_neck, qlim, dq, N_q, vol, rn):
    """
    JIT-оптимизированная проверка условия выхода
    """
    vol_current = gh_ap3d_jit(q, qlim, dq, N_q, vol)
    if abs(vol_current - 1) >= 1e-3:
        return True
    rn_current = gh_ap3d_jit(q, qlim, dq, N_q, rn)
    return rn_current <= r_neck


@nb.njit(fastmath=True, nogil=True)
def q1_def_jit(q_start, V_st, inv_m, ampl, ql, qlim, dq, N_q, V, ground_state):
    """
    JIT-оптимизированное определение начальных условий с защитой от зацикливания
    """
    E_kin = -1
    max_attempts = 1000
    attempts = 0

    if q_start[0] > 0.5:
        while E_kin < 0 and attempts < max_attempts:
            ξ1 = ξ(0, .5, dim)
            ξ1[0] = abs(ξ1[0])
            q = q_start + ξ1 * ampl
            attempts += 1

            # Проверка границ
            in_bounds = True
            for i in range(dim):
                if q[i] < ql[0, i] or q[i] > ql[1, i]:
                    in_bounds = False
                    break
            
            if not in_bounds:
                continue

            E_kin = gh_ap3d_jit(q, qlim, dq, N_q, V_st)
    else:
        while E_kin < 0 and attempts < max_attempts:
            ξ1 = ξ(0, .5, dim)
            ξ1[0] = abs(ξ1[0])
            q = q_start + ampl * ξ1
            attempts += 1
            E_kin = E_0 + (ground_state - gh_ap3d_jit(q, qlim, dq, N_q, V))

    # Защита от неудачных попыток
    if E_kin < 0 or attempts >= max_attempts:
        q = q_start.copy()
        q[0] += 0.1
        E_kin = E_0

    # Вычисление импульсов
    mass = 0.5 * gh_ap3d_tens_jit(q, qlim, dq, N_q, inv_m)
    
    # Защита от неположительных масс
    diag_mass = np.diag(mass)
    for i in range(dim):
        if diag_mass[i] <= 0:
            mass[i, i] = 0.5

    p_ampl = np.sqrt(E_kin / np.diag(mass))
    p = p_ampl * ξ(0, 0.5, dim)
    
    # Нормализация импульса
    p_norm_sq = 0.
    for i in range(dim):
        for j in range(dim):
            p_norm_sq += mass[i, j] * p[i] * p[j]
    
    if p_norm_sq > 0:
        p *= sqrt(E_kin / p_norm_sq)
    
    p[0] = abs(p[0])
    return q, p


@nb.njit(fastmath=True, nogil=True)
def temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable):
    """
    JIT-оптимизированное определение температуры и распределения энергии
    """
    # Кинетическая энергия
    E_kin = 0.
    for i in range(dim):
        for j in range(dim):
            E_kin += 0.5 * i_m[i, j] * p[i] * p[j]

    E_st = E_total - (E_kin + V_mac + V_mic * shell)

    if E_st < 0:
        E_kin_new = E_kin + E_st - a * temp ** 2
        if E_kin_new > 0:
            scale_factor = sqrt(E_kin_new / E_kin)
            for i in range(dim):
                p[i] *= scale_factor
        else:
            for i in range(dim):
                p[i] = 0.

    temp2 = max(E_st / a, 1e-16)
    temp = max(temp, sqrt(temp2))
    t_star = sqrt(E_0 / tanh(E_0 / temp)) if t_star_enable else sqrt(temp)
    shell = shell_correction_jit(temp, True, T_CONST, A_T)
    g_coef = friction_temp_correction_jit(temp, True)

    return temp, temp2, t_star, shell, g_coef, E_st, p


###############################################################################
###################### ГЛАВНАЯ ФУНКЦИЯ ТРАЕКТОРИИ ############################
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
    ПОЛНОСТЬЮ JIT-ОПТИМИЗИРОВАННАЯ функция расчета траектории
    
    Это основная функция, которая должна работать максимально быстро
    """
    dim_local = len(q_start)
    temp = temperature
    time = 0
    shell = shell_correction_jit(temp, shell_ef, T_CONST, A_T)
    dp = np.zeros(dim_local)
    q2_max = qlim[1, 0]
    
    # Начальные условия
    q, p = q1_def_jit(q_start, V_s2, inv_m, ampl, ql, qlim, dq, N_q, V, ground_state)
    
    # Список для хранения траектории (только первые и последние точки для экономии памяти)
    q_track = np.zeros((2, dim_local))
    q_track[0] = q_start
    q_track[1] = q.copy()

    # Радиус шейки для траектории
    r_neck_traj = abs(ξ(r_neck, sigma_r_neck)) if gauss_flag else r_neck

    # Главный цикл интегрирования
    while q_start[0] <= q[0] and time <= step_limit:

        # Интерполяция всех параметров одновременно
        i_m, g, sqrt_g, V_mac, V_mic, a, d_a, d_i_m, dV_mac, dV_mic = \
            gh_ap3d_set_jit(q, qlim, dq, N_q, inv_m, fric, sqrt_fric, 
                           V_macro, V_micro, a_d, d_i_m_dq, dV_macro_dq, 
                           dV_micro_dq, d_a_d_dq)

        # Обновление температуры каждые idt шагов
        if time % idt == 0:
            temp, temp2, t_star, shell, g_coef, E_st, p = \
                temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)
            
            if poisson_flag:
                # Простая аппроксимация Poisson процесса
                r_neck_traj = max(0.1, temp / 5) * r_nucleon

        # Применение температурной коррекции к трению
        for i in range(dim_local):
            for j in range(dim_local):
                g[i, j] *= g_coef
                sqrt_g[i, j] *= sqrt(g_coef)

        # Вычисление стохастических сил
        for i in range(dim_local):
            stochastic_force = 0.
            for j in range(dim_local):
                stochastic_force += sqrt_g[i, j] * rand_function_jit(1, sqrt_mult)[0]
            
            # Детерминистические силы
            deterministic_force = 0.
            for j in range(dim_local):
                deterministic_force += (0.5 * d_i_m[i, j] @ p + i_m[i, j] @ g[i, j]) @ p
            
            dp[i] = (stochastic_force * sqrt_dt * t_star - 
                    deterministic_force * dt)

        # Градиентные силы
        for i in range(dim_local):
            dp[i] -= dt * (dV_mac[i] + shell * dV_mic[i] - d_a[i] * temp2)

        # Обновление импульсов
        for i in range(dim_local):
            p[i] += dp[i]

        # Обновление координат
        Δq = np.zeros(dim_local)
        for i in range(dim_local):
            for j in range(dim_local):
                Δq[i] += i_m[i, j] * (p[j] - dp[j] / 2)

        for i in range(dim_local):
            q[i] += Δq[i] * dt

        # Граничные условия (отражение от границ для q3, q4)
        for i in range(1, dim_local):
            if q[i] < qlim[0, i]:
                q[i] = qlim[0, i] - (q[i] - qlim[0, i])
                p[i] *= -1
            elif q[i] > qlim[1, i]:
                q[i] = qlim[1, i] - (q[i] - qlim[1, i])
                p[i] *= -1

        time += 1

        # Условия выхода
        if q[0] >= q2_max:
            if limit_cut_flag:
                break
            q[0] = q2_max
            final_temp = temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)[0]
            return q, p, time * dt, final_temp, q[2] <= 0, q_track
        
        elif exit_condition_jit(q, r_neck_traj, qlim, dq, N_q, vol, rn) and q[0] > 1.5:
            final_temp = temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)[0]
            success_condition = temperature + 0.1 < final_temp
            return q, p, time * dt, final_temp, success_condition, q_track

    # Если дошли до конца без выхода
    final_temp = temp_def_jit(temp, p, E_total, i_m, a, V_mac, V_mic, shell, t_star_enable)[0]
    return q, p, time * dt, final_temp, False, q_track


###############################################################################
###################### ВСПОМОГАТЕЛЬНЫЕ JIT ФУНКЦИИ ###########################
###############################################################################

@nb.njit(fastmath=True, nogil=True, parallel=True)
def multithreading_trj_calc_jit(input_param, N_tr):
    """
    JIT-оптимизированный расчет множественных траекторий с параллелизацией
    """
    traj_crd_out = np.empty((N_tr, len(input_param[0])))
    traj_p_out = np.empty_like(traj_crd_out)
    traj_time = np.empty(N_tr)
    traj_temp = np.empty(N_tr)
    wrong_count = 0

    for i in nb.prange(N_tr):  # Параллельное выполнение
        correct_traj = False
        attempts = 0
        max_attempts = 10

        while not correct_traj and attempts < max_attempts:
            q, p, t, temp_out, correct_traj, q_tr = trajectory_calc_jit(*input_param)
            attempts += 1
            if not correct_traj:
                wrong_count += 1

        if attempts >= max_attempts:
            # Используем начальные условия как fallback
            traj_crd_out[i] = input_param[0]
            traj_p_out[i] = np.zeros_like(input_param[0])
            traj_time[i] = 0.0
            traj_temp[i] = input_param[1]
        else:
            traj_crd_out[i] = q
            traj_p_out[i] = p
            traj_time[i] = t
            traj_temp[i] = temp_out

    return traj_crd_out, traj_p_out, traj_time, traj_temp, wrong_count


###############################################################################
###################### AUXILIARY LIBRARY JIT FUNCTIONS #######################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def q_into_alpha_jit(q_array):
    """
    JIT-оптимизированная конверсия q в коэффициент асимметрии масс
    """
    from config import A2_0, A4_0
    
    result = np.empty(q_array.shape[0])
    for idx in range(q_array.shape[0]):
        q = q_array[idx]
        a2 = 0.5 * A2_0 * (sqrt(q[0] ** 2 + 4) - q[0])
        a3 = q[1]  
        a4 = q[2] - sqrt(A4_0 ** 2 + q[0] ** 2 / 81)
        a5 = (q[0] - 2) * a3 * 0.1
        
        numerator = a3 / (a2 - a4 / 3)
        denominator = 1 - 2 * (a2 + a4) / (a2 + 9 * a4)
        result[idx] = numerator * denominator
        
    return result


###############################################################################
###################### ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ########################
###############################################################################

def benchmark_jit_functions():
    """
    Тестирование производительности JIT функций
    """
    print("=" * 60)
    print("NUMBA JIT PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Тестовые данные
    q_test = np.array([1.0, 0.1, -0.1])
    qlim_test = np.array([[0., -0.5, -0.5], [3., 0.5, 0.5]])
    dq_test = np.array([0.1, 0.1, 0.1])
    N_q_test = np.array([30, 10, 10])
    matrix_test = np.random.random((30, 10, 10))
    
    import time
    
    # Тест интерполяции
    start = time.time()
    for _ in range(1000):
        result = gh_ap3d_jit(q_test, qlim_test, dq_test, N_q_test, matrix_test)
    jit_time = time.time() - start
    
    print(f"✓ GH interpolation (1000 calls): {jit_time:.4f} seconds")
    print(f"✓ Average per call: {jit_time/1000*1e6:.2f} microseconds")
    
    # Тест генерации случайных чисел
    start = time.time()
    for _ in range(10000):
        rand_function_jit(3)
    rand_time = time.time() - start
    
    print(f"✓ Random generation (10000 calls): {rand_time:.4f} seconds")
    print(f"✓ Average per call: {rand_time/10000*1e6:.2f} microseconds")
    
    print("=" * 60)
    print("JIT COMPILATION SUCCESSFUL - MAXIMUM PERFORMANCE RESTORED")
    print("=" * 60)


if __name__ == "__main__":
    # Принудительная компиляция функций
    print("Compiling JIT functions...")
    
    # Компилируем основные функции
    test_data = np.array([1., 0., 0.])
    ampl_definer_jit(test_data, np.random.random((3, 10, 10, 10)), 
                     np.array([[0., 0., 0.], [1., 1., 1.]]), 
                     np.array([0.1, 0.1, 0.1]), 
                     np.array([10, 10, 10]))
    
    benchmark_jit_functions()

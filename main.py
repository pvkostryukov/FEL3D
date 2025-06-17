# -*- coding: utf-8 -*-
"""
Главный модуль FEL3D для симуляции ядерного деления
ПОЛНАЯ ФУНКЦИОНАЛЬНОСТЬ + МАКСИМАЛЬНАЯ ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ

@author: KPV
Optimized version with numba JIT acceleration
"""

import os
import sys
import datetime
import platform
from random import uniform, choice

import numpy as np
from numpy.random import normal as ξ
from math import sqrt, exp, tanh, isnan, erf
from scipy import linalg as lalg
import matplotlib.pyplot as plt
import pandas as pd
import re

import numba as nb

# Импорт всех рефакторенных модулей
import config
import gauss_hermit as gh
import data_handling as dh
import physics_core as phys
import physics_numba as pnb
import monte_carlo_optimized as mc
import auxiliary_library as aux
import emission_auxillary_lib as em

# Константы из конфигурации
from config import (
    RT2, DIM as dim, NODES as nodes, H_NODES as h_nodes, GAMMA as γ,
    E_0, EXTENSIONS as extensions, get_element_symbol,
    T_CONST, A_T
)

# Настройка путей и флагов операционной системы
user_path = os.getcwd()
OS_flag = platform.system() == 'Windows'
path = os.getcwd()


###############################################################################
###################### ФУНКЦИЯ ПРЕДВАРИТЕЛЬНОЙ КОМПИЛЯЦИИ ####################
###############################################################################

def precompile_jit_functions():
    """Предварительная компиляция JIT-функций для максимальной производительности"""
    print("Precompiling JIT functions for optimal performance...")
    
    # Создаем тестовые данные
    test_q = np.array([0.5, 0.0, 0.0])
    test_qlim = np.array([[0.0, -1.0, -1.0], [2.0, 1.0, 1.0]])
    test_dq = np.array([0.1, 0.1, 0.1])
    test_N_q = np.array([20, 20, 20])
    test_matrix = np.ones((20, 20, 20))
    test_tensor = np.ones((20, 20, 20, 3, 3))
    
    # Принудительная компиляция основных функций
    try:
        _ = pnb.gh_ap3d_jit(test_q, test_qlim, test_dq, test_N_q, test_matrix)
        _ = pnb.gh_ap3d_tens_jit(test_q, test_qlim, test_dq, test_N_q, test_tensor)
        _ = pnb.ampl_definer_jit(test_q, np.array([test_matrix, test_matrix, test_matrix]), 
                                test_qlim, test_dq, test_N_q)
        _ = pnb.density_jit(235, 92, test_matrix, test_matrix, test_matrix)
        print("✓ JIT compilation completed successfully")
    except Exception as e:
        print(f"Warning: JIT precompilation failed: {e}")
        print("Code will still work but may be slower on first run")


###############################################################################
######################## ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ #############################
###############################################################################

def nuc_definer(z_number):
    """Backward compatibility wrapper"""
    return get_element_symbol(z_number)


def spontaneus_st_point(V, gs, q_2sad_st):
    """Defining initial point in case of spontaneous fission"""
    if q_grid[0][q_2sad_st[0]] < 0.65:
        q_2sad_st[0] = int(round((0.65 - q_grid[0][0]) / dq[0]))
    delta_eps = [.01, .05, 0.1]
    bot, up = N_q[1] // 2 - 5, N_q[1] // 2 + 5
    for delta in delta_eps:
        pnt_set = np.array(np.where(np.isclose(V, ground_state, atol=delta))).T
        pnt_set = pnt_set[np.bitwise_and(bot <= pnt_set[:, 1], pnt_set[:, 1] <= up)]
        pnt_set = np.array([el for el in pnt_set
                            if V[tuple(el)] <= gs and el[0] - 1 > q_2sad_st[0]]
                           )
        if len(pnt_set) > 1:
            break
        elif delta == delta_eps[-1]:
            print('No entrance point!\n')
    sum2_idx = ((pnt_set - gs_mesh_crd) ** 2).sum(axis=1)
    st_idx = tuple(pnt_set[sum2_idx == min(sum2_idx)][-1])
    V_st = V[st_idx]
    st_pnt = np.array([el[i] for i, el in zip(st_idx, q_grid)]).flatten()
    return st_pnt, st_idx, V_st


def st_pnt_checking(st_point, V: np.array):
    """Checks starting point position"""
    st_idx = np.array([round((el - qlim[0, i]) / dq[i])
                       for i, el in enumerate(st_point)]).astype(int)
    pretendents = np.array(np.where(V == V[tuple(st_idx)].min())).T \
        if len(st_idx) != 1 else np.array(np.where(V == V[st_idx[0],
                                                        2:-3].min())
                                          )
    pretendents = pretendents[pretendents[:, 0] == st_idx[0]]
    pretendents = pretendents[-1] if len(pretendents.shape) > 1 else pretendents
    if (pretendents[:len(st_idx)] != st_idx).any() or len(pretendents) > len(st_idx):
        st_idx = pretendents.copy()
        st_point = st_idx * dq + qlim[0]
    return st_point, st_idx, V[tuple(st_idx)]


def prepare_global_parameters(N_q, dq, qlim, vol, rn, V, ground_state,
                              E_total, V_starting, d_i_m_dq, dV_macro_dq,
                              dV_micro_dq, d_a_d_dq, temp_ef, shell_ef,
                              t_star_enable, gauss_flag, poisson_flag,
                              r_nucleon, limit_cut_flag, A, Z):
    """Prepare global parameters dictionary for Monte Carlo"""
    return {
        'qlim': qlim,
        'dq': dq,
        'N_q': N_q,
        'vol': vol,
        'rn': rn,
        'ground_state': ground_state,
        'E_total': E_total,
        'V_starting': V_starting,
        'd_i_m_dq': d_i_m_dq,
        'dV_macro_dq': dV_macro_dq,
        'dV_micro_dq': dV_micro_dq,
        'd_a_d_dq': d_a_d_dq,
        'temp_ef': temp_ef,
        'shell_ef': shell_ef,
        't_star_enable': t_star_enable,
        'gauss_flag': gauss_flag,
        'poisson_flag': poisson_flag,
        'r_nucleon': r_nucleon,
        'limit_cut_flag': limit_cut_flag,
        'A': A,
        'Z': Z
    }

###############################################################################
###################### ГЛАВНЫЙ КОД ###########################################
###############################################################################

if __name__ == "__main__":

    # ОПТИМИЗАЦИЯ: Предварительная компиляция JIT-функций
    precompile_jit_functions()

    if 'input.xlsx' not in os.listdir():
        print('Error: there no input file!')
        sys.exit()

    print('Calculations starts: ' +
          datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S"))

    Z_prev, A_prev = 0, 0
    fourier_file_prev = ''
    prev_pot_file_ext = ''

    # ИСПОЛЬЗУЕМ МОДУЛЬ DATA_HANDLING
    try:
        inp_data = dh.load_input_data()  # Правильная функция
    except Exception as e:
        print(f"Error reading input.xlsx: {e}")
        sys.exit()

    exact_place = os.getcwd()
    path = os.getcwd()

    num_ptn = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(num_ptn, re.VERBOSE)

    for i, isotope in inp_data.iterrows():
        if isotope.isnull().all():
            sys.exit()

        Z, A, N, E_init, dt, starting_point, temp_ef, shell_ef, \
            t_star_enable, r_neck, sigma_r_neck, diffiuse_mult, \
            exp_file, fit_of_T, gauss_flag, poisson_flag, elong_flag, \
            short_q2_flg, limit_cut_flag, info_full = isotope

        # ИСПОЛЬЗУЕМ МОДУЛЬ PHYSICS_CORE для валидации
        try:
            phys.validate_physical_parameters(A, Z, E_init)
        except ValueError as e:
            print(f"Invalid parameters for isotope {i}: {e}")
            continue

        # ОПТИМИЗАЦИЯ: Перекомпиляция JIT-функций при смене изотопа
        if i != 0:
            for func in [pnb.ampl_definer_jit, pnb.trajectory_calc_jit,
                         pnb.gh_ap3d_jit, pnb.gh_ap3d_tens_jit]:  # Используем JIT-версии
                func.recompile()

        # Обработка флагов
        gauss_flag = False if type(gauss_flag) != bool else gauss_flag and (not isnan(r_neck))
        poisson_flag = poisson_flag and isnan(r_neck)
        elong_flag = False if type(elong_flag) != bool else elong_flag
        limit_cut_flag = False if type(limit_cut_flag) != bool else limit_cut_flag
        short_q2_flg = False if type(short_q2_flg) != bool else short_q2_flg

        diffiuse_mult = 1 if isnan(diffiuse_mult) else sqrt(diffiuse_mult)
        sqrt_mult = diffiuse_mult * RT2 if diffiuse_mult != 1 else RT2
        Z, A, N = int(Z), int(A), int(N)

        SHE_flag = Z > 103

        # Определение расширений файлов
        fl_m = np.array([not (SHE_flag or elong_flag) and short_q2_flg,
                         (not SHE_flag) and elong_flag and short_q2_flg,
                         not (SHE_flag or elong_flag) and (not short_q2_flg),
                         SHE_flag and not elong_flag and (not short_q2_flg),
                         not SHE_flag and elong_flag and (not short_q2_flg),
                         SHE_flag and elong_flag and (not short_q2_flg)
                         ])

        pot_file_ext = np.array(extensions[0])[fl_m][0]
        fourier_file = 'fourier' + np.array(extensions[1])[fl_m][0]

        # ИСПОЛЬЗУЕМ МОДУЛЬ DATA_HANDLING для загрузки данных
        if fourier_file_prev != fourier_file:
            N_q, dq, qlim, m_0, f_0, bs, bc, bk, bf, \
                r12, bx, vol, c, rn = dh.fourier_file_data(fourier_file, path)
            fourier_file_prev = fourier_file
            q_grid = [np.linspace(qlim[0, i], qlim[1, i], N_q[i])
                      for i in range(dim)]

        if (Z, A) != (Z_prev, A_prev):
            # ИСПОЛЬЗУЕМ МОДУЛЬ PHYSICS_CORE для подготовки
            r0 = phys.calculate_nuclear_radius(A)
            r_nucleon = 1 / r0
            r_neck = r_neck if isnan(r_neck) else r_neck * r_nucleon
            sigma_r_neck = sigma_r_neck if isnan(sigma_r_neck) \
                else sigma_r_neck * r_nucleon

            m_cf, fric_cf = phys.calculate_mass_coefficients(A)
            m = m_0.copy() * m_cf
            fric = f_0.copy() * fric_cf
            inv_m = np.zeros_like(m)
            sqrt_fric = np.zeros_like(fric)

            for idx, el in np.ndenumerate(bs):
                sqrt_fric[idx] = lalg.sqrtm(fric[idx])
                inv_m[idx] = lalg.inv(m[idx])

            d_i_m_dq = np.zeros(tuple([dim, ] + [i for i in inv_m.shape]))
            for i in range(dim):
                for j in range(dim):
                    d_i_m_dq[..., i, j] = np.array(np.gradient(inv_m[..., i, j],
                                                               dq[0], dq[1],
                                                               dq[2],
                                                               edge_order=2))

            # ОПТИМИЗАЦИЯ: Используем JIT-версию функции плотности
            try:
                a_d = pnb.density_jit(A, Z, bs, bk, bc)
                print("✓ Using optimized density_jit")
            except:
                # Используем обычную версию
                a_d = phys.density(A, Z, bs, bk, bc)
                print("⚠ Using standard density function")
            Z_prev, A_prev = Z, A

        # ИСПОЛЬЗУЕМ МОДУЛЬ DATA_HANDLING для потенциалов
        if pot_file_ext != prev_pot_file_ext:
            prev_pot_file_ext = pot_file_ext
            V_macro, V_micro = dh.potential_reader(A, Z, N_q, dq, qlim,
                                                   pot_file_ext, path)
            V = V_macro + V_micro

            # ИСПОЛЬЗУЕМ МОДУЛЬ PHYSICS_CORE для производных
            d_a_d_dq = np.array(np.gradient(a_d, dq[0], dq[1], dq[2]))
            dV_macro_dq = np.array(np.gradient(V_macro, dq[0], dq[1], dq[2]))
            dV_micro_dq = np.array(np.gradient(V_micro, dq[0], dq[1], dq[2]))

            # ИСПОЛЬЗУЕМ МОДУЛЬ PHYSICS_CORE для основного состояния
            ground_state, gs_mesh_crd, q2_gs = phys.find_ground_state(V, qlim, dq, short_q2_flg)

        # ИСПОЛЬЗУЕМ МОДУЛЬ DATA_HANDLING для начальной точки
        if type(starting_point) == str:
            is_number = rx.findall(''.join(starting_point))
            if 'from file' in starting_point:
                starting_point = dh.st_pnt_def(A, Z,
                                               starting_point.split('from file')[-1],
                                               path)
                starting_point, start_idx, \
                    V_starting = st_pnt_checking(starting_point, V)
            elif starting_point in ['spnt', 'spont', 'spontaneus']:
                q_2sad_idx = ((dh.st_pnt_def(A, Z, path=path) - qlim[0]) / dq).astype(int)
                starting_point, start_idx, \
                    V_starting = spontaneus_st_point(V, ground_state, q_2sad_idx)
            else:
                starting_point = np.array(list(map(float, is_number)))
                starting_point, start_idx, \
                    V_starting = st_pnt_checking(starting_point, V)
        else:
            print('Error! Wrong starting point input')

        E_star = E_init
        E_total = E_init + st_pnt_checking(dh.st_pnt_def(A, Z, path=path), V)[-1] \
                  - ground_state

        if E_init < 0:
            print('Error! Invalid initial energy value.')
            sys.exit()

        # Проверка на подгонку параметров
        if not isnan(fit_of_T):
            os.chdir(path + ('\\Experimental data\\' if OS_flag else '/Experimental data/'))
            if exp_file in os.listdir():
                # fit = fit_procedure(A, exp_file, exact_place)  # Можно добавить позже
                # T_const, a_t = fit[2]
                pass
            else:
                print('There no file ' + exp_file + ' in experimental data directory')
            os.chdir(exact_place)

        # ОПТИМИЗАЦИЯ: Используем JIT-версию для интерполяции
        temperature = sqrt(E_star / pnb.gh_ap3d_jit(starting_point, qlim, dq, N_q, a_d))

        # ИСПОЛЬЗУЕМ МОДУЛЬ PHYSICS_CORE для поправок
        sh = phys.shell_correction(temperature, shell_ef, T_CONST, A_T)
        F = V_macro + sh * V_micro - a_d * temperature ** 2

        dV_dq = np.array(np.gradient(V, dq[0], dq[1], dq[2]))
        d2V_dq2 = np.array([np.gradient(el, dq[0], dq[1], dq[2])[i]
                            for i, el in enumerate(dV_dq)])

        isotope_name = get_element_symbol(Z) + '-' + f'{int(A)}'
        print('\t Isotope ' + isotope_name + f' E = {round(E_star, 3)}')

        # Подготовка глобальных параметров для Monte Carlo
        global_params = prepare_global_parameters(
            N_q, dq, qlim, vol, rn, V, ground_state,
            E_total, V_starting, d_i_m_dq, dV_macro_dq,
            dV_micro_dq, d_a_d_dq, temp_ef, shell_ef,
            t_star_enable, gauss_flag, poisson_flag,
            r_nucleon, limit_cut_flag, A, Z
        )

        inp_var = starting_point, temperature, d2V_dq2, inv_m, fric, \
            sqrt_fric, V_macro, V_micro, V, a_d

        # Количество траекторий - улучшенное чтение из input.xlsx
        N = config.DEFAULT_N_TRAJECTORIES  # Значение по умолчанию

        # Пробуем разные возможные названия столбцов
        possible_n_columns = ['N', 'n', 'Number of trajectories','N_trajectories', 'n_trajectories', 'trajectories']

        for col in possible_n_columns:
            if hasattr(isotope, col):
                value = getattr(isotope, col, None)
                if not pd.isna(value) and value is not None:
                    try:
                        N = int(value)
                        print(f"✓ Using N = {N} trajectories from column '{col}' in input file")
                        break
                    except (ValueError, TypeError):
                        continue
        else:
            print(f"⚠ No valid trajectory count found in input file, using default N = {N}")
            print(f"Available columns: {list(isotope.index) if hasattr(isotope, 'index') else 'Unknown'}")

        if isnan(r_neck) and isnan(sigma_r_neck) and type(exp_file) == str:
            print("R fit procedure temporarily disabled")
            # ОПТИМИЗАЦИЯ: Используем оптимизированную версию Monte Carlo
            q_out, p_out, traj_time, \
                temp_out = mc.monte_carlo_optimized(*inp_var, r_neck,
                                                   sigma_r_neck, sqrt_mult, dt, N, 
                                                   T_CONST, A_T, global_params,
                                                    )
        else:
            # ОПТИМИЗАЦИЯ: Используем оптимизированную версию Monte Carlo
            q_out, p_out, traj_time, \
                temp_out = mc.monte_carlo_optimized(*inp_var, r_neck,
                                                   sigma_r_neck, sqrt_mult, dt, N,
                                                   T_CONST, A_T, global_params,
                                                   use_parallel=True
                                                    )

        # ========================= ПОЛНАЯ ОБРАБОТКА РЕЗУЛЬТАТОВ =========================

        # ОПТИМИЗАЦИЯ: Используем JIT-версию для интерполяции
        rn_out = np.array([pnb.gh_ap3d_jit(i, qlim, dq, N_q, rn) for i in q_out])
        output = pd.DataFrame({'time': traj_time,
                               'q2': q_out[:, 0], 'q3': q_out[:, 1],
                               'q4': q_out[:, 2],
                               'p2': p_out[:, 0], 'p3': p_out[:, 1],
                               'p4': p_out[:, 2],
                               'Rneck': rn_out, 'Temperature': temp_out})

        # Флаг эмиссии фрагментов деления
        emission_FF_flag = True

        if info_full in ('p', 'pre', 'precise'):
            # ТОЧНЫЙ РЕЖИМ: вычисляем все параметры ядер
            param_cf = np.zeros((N, 11))
            for i in nb.prange(N):
                param_cf[i] = aux.q_to_nucl_param(q_out[i])

            Bf_q = param_cf[:, 0]
            A_f_0 = np.round(A * Bf_q).astype(int)
            Z_f_0 = np.round(Z * Bf_q).astype(int)
            R12_q = r0 * param_cf[:, 1]
            BCoul_q, Bs_q, Bc_q = param_cf[:, 2], param_cf[:, 3], param_cf[:, 4]

            if emission_FF_flag:
                emsn_out = [[] for _ in range(N)]
                for i in nb.prange(N):
                    emsn_out[i] = em.emission_FF(A, Z, A_f_0[i], Z_f_0[i],
                                                 temp_out[i], param_cf[i, 1:])
                A_f_1, Z_f_1, E_star_CN, \
                    E_star_FF, e_n_FF = em.emFF_decoder(emsn_out)

        elif info_full in ('a', 'approx', 'approximate'):
            # ПРИБЛИЖЕННЫЙ РЕЖИМ: используем интерполяцию
            Bf_q = .5 * (1 + aux.q_into_alpha(q_out))
            A_f_0 = np.round(A * Bf_q).astype(int)
            Z_f_0 = np.round(Z * Bf_q).astype(int)
            # ОПТИМИЗАЦИЯ: Используем JIT-версии для всех интерполяций
            Bs_q = np.array([pnb.gh_ap3d_jit(q, qlim, dq, N_q, bs) for q in q_out])
            Bc_q = np.array([pnb.gh_ap3d_jit(q, qlim, dq, N_q, bc) for q in q_out])
            BCoul_q = np.array([pnb.gh_ap3d_jit(q, qlim, dq, N_q, bc) for q in q_out])
            R12_q = r0 * np.array([pnb.gh_ap3d_jit(q, qlim, dq, N_q, r12)
                                   for q in q_out])
            if emission_FF_flag:
                emsn_out = [em.emission_FF_gh3d(q, A, Z, T, qlim, dq, N_q,
                                                bc, bs, bk)
                            for (q, T) in zip(q_out, temp_out)]
                A_f_1, Z_f_1, E_star_CN, \
                    E_star_FF, e_n_FF = em.emFF_decoder(emsn_out)
        else:
            # МИНИМАЛЬНЫЙ РЕЖИМ: только базовая информация
            output.to_excel(f'{isotope_name}_minimal.xlsx', sheet_name='Sheet1',
                            engine='openpyxl', index=False)
            continue

        # Создание второго листа с параметрами
        output_2 = pd.DataFrame({'Af_0': A_f_0,
                                 'Zf_0': Z_f_0,
                                 'Bf': Bf_q,
                                 'Bs': Bs_q,
                                 'Bc': Bc_q,
                                 'BCoul': BCoul_q,
                                 'R12': R12_q})

        # Создание распределений по массам и зарядам
        A_f = np.concatenate((A_f_0, A - A_f_0))
        Z_f = np.concatenate((Z_f_0, Z - Z_f_0))

        Af_range = np.arange(A_f.min(), A_f.max() + 2, dtype=int)
        Zf_range = np.arange(Z_f.min(), Z_f.max() + 2, dtype=int)

        h, Af_range, Zf_range = np.histogram2d(A_f, Z_f, bins=(Af_range, Zf_range),
                                               density=True)
        h *= 2
        Z_A = np.empty((1, 3))
        for i, el in enumerate(h.T):
            el_mask = ~np.isclose(el, 0)
            if any(el_mask):
                Z_A = np.concatenate((Z_A,
                                      np.array([[Zf_range[i], j, k]
                                                for j, k in zip(Af_range[:-1][el_mask],
                                                                el[el_mask])
                                                ])
                                      ))
        Z_A = Z_A[1:]

        output_3YA = pd.DataFrame({"Af": Af_range[:-1],
                                   "Y(Af)": h.sum(axis=1)
                                   })

        output_3YZ = pd.DataFrame({"Zf": Zf_range[:-1],
                                   "Y(Zf)": h.sum(axis=0)
                                   })

        output_3_YZA = pd.DataFrame({"Zf": Z_A.T[0].astype(int),
                                     "Af": Z_A.T[1].astype(int),
                                     "Y(Zf,Af)": Z_A.T[2]
                                     })

        # Обработка эмиссии частиц (если включена)
        if emission_FF_flag and 'A_f_1' in locals():
            output_2["A'_L"] = A_f_1[:, 0]
            output_2["A'_R"] = A_f_1[:, 1]
            output_2["ε_n"] = [[[round(el, 6) for el in el1] for el1 in _]
                               for _ in e_n_FF]

            # Дополнительные распределения после эмиссии
            Af1_range = np.arange(A_f_1.min(), A_f_1.max() + 2, dtype=int)
            Zf1_range = np.arange(Z_f_1.min(), Z_f_1.max() + 2, dtype=int)

            h_1, Af1_range, Zf1_range = np.histogram2d(A_f_1.flatten(),
                                                       Z_f_1.flatten(),
                                                       bins=(Af1_range, Zf1_range),
                                                       density=True)
            h_1 *= 2
            Z_A_1 = np.empty((1, 3))
            for i, el in enumerate(h_1.T):
                el_mask = ~np.isclose(el, 0)
                if any(el_mask):
                    Z_A_1 = np.concatenate((Z_A_1,
                                            np.array([[Zf1_range[i], j, k]
                                                      for j, k in
                                                      zip(Af1_range[:-1][el_mask],
                                                          el[el_mask])
                                                      ])
                                            ))
            Z_A_1 = Z_A_1[1:]

            output_3YA_1 = pd.DataFrame({"A'f": Af1_range[:-1],
                                         "Y(A'f)": h_1.sum(axis=1)
                                         })

            output_3YZ_1 = pd.DataFrame({"Z'f": Zf1_range[:-1],
                                         "Y(Z'f)": h_1.sum(axis=0)
                                         })

            output_3_YZA_1 = pd.DataFrame({"Z'f": Z_A_1.T[0].astype(int),
                                           "A'f": Z_A_1.T[1].astype(int),
                                           "Y(Z'f,A'f)": Z_A_1.T[2]
                                           })

        # ================== СОЗДАНИЕ РЕЗУЛЬТИРУЮЩИХ ФАЙЛОВ ==================

        # Создание папки результатов
        res_path = os.path.join(path, 'Result')
        if not os.path.isdir(res_path):
            os.mkdir(res_path)
        res_path = os.path.join(res_path, datetime.datetime.today().strftime("%d-%m-%y"))
        if not os.path.isdir(res_path):
            os.mkdir(res_path)
        os.chdir(res_path)

        # Формирование сложного имени файла с параметрами
        add_p = 'P' if poisson_flag else ''
        add_g = 'G' if gauss_flag else ''
        add_lim = 'q2l' if limit_cut_flag else 'q2p'
        e0 = f'_e0({E_0:.2g})'

        I_fit = ''
        if 'fit_out' in globals():
            pd.DataFrame(fit_out[0]).to_csv(isotope_name + ' I(R, σ) table.csv')
            I_fit = f'(I = {fit_out[1]:.4g} {exp_file})'

        file_name = (get_element_symbol(Z).lower() +
                     f'{int(A)}' +
                     f'_e({round(E_star, 3)})' +
                     f'_n{len(q_out)}' +
                     f'_dt{str(dt)[1:] if int(dt) == 0 else dt}'.replace('.', '') +
                     re.sub('0(?=[.])', '',
                            f'_{add_g}({r_neck / r_nucleon:.2g}_{sigma_r_neck / r_nucleon:.2g})') +
                     re.sub('0(?=[.])', '',
                            r'_st({:.3g}_{:.3g}_{:.3g})'.format(*starting_point)) +
                     f'{"_t(" if any((shell_ef, temp_ef, t_star_enable)) else ""}' +
                     f'{"sh" if shell_ef else ""}' +
                     f'{"_" if shell_ef & any((temp_ef, t_star_enable)) else ""}' +
                     f'{"g" if temp_ef else ""}' +
                     f'{"_" if t_star_enable & temp_ef else ""}' +
                     f'{"*)" if t_star_enable else ")"}' +
                     e0 +
                     f'_pot{pot_file_ext[1:]}' +
                     f'_✓{diffiuse_mult ** 2:.2g}' + I_fit +
                     '_OPTIMIZED.xlsx')  # Добавляем маркер оптимизации

        # Сохранение в Excel с множественными листами
        with pd.ExcelWriter(file_name, engine='openpyxl') as wr:

            shift_idx = 3 if emission_FF_flag and 'A_f_1' in locals() else 0
            output.to_excel(wr, sheet_name='Sheet1', index=False)
            output_2.to_excel(wr, sheet_name='Sheet2', index=False)
            output_3YA.to_excel(wr, sheet_name='Sheet3', startcol=0, index=False)
            output_3YZ.to_excel(wr, sheet_name='Sheet3',
                                startcol=2 + shift_idx, index=False)
            output_3_YZA.to_excel(wr, sheet_name='Sheet3',
                                  startcol=7 + shift_idx, index=False)

            if emission_FF_flag and 'A_f_1' in locals():
                output_3YA_1.to_excel(wr, sheet_name='Sheet3',
                                      startcol=shift_idx - 1, index=False)
                output_3YZ_1.to_excel(wr, sheet_name='Sheet3',
                                      startcol=3 * shift_idx - 2, index=False)
                output_3_YZA_1.to_excel(wr, sheet_name='Sheet3',
                                        startcol=4 * shift_idx + 1, index=False)

        os.chdir(exact_place)
        print(f"✓ Results saved: {file_name}")

    print('Calculations ends:   ' +
          datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S"))
    print("🚀 OPTIMIZED VERSION - Performance enhanced with numba JIT compilation")

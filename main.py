# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:20:49 2023

@author: KPV 
"""

###############################################################################
##################### CALLING USED LIBRARIES & PACKAGES #######################
###############################################################################

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
from tqdm import tqdm
from numba_progress import ProgressBar
import pandas as pd
import re

import numba as nb
import auxiliary_library as aux
import emission_auxillary_lib as em 


###############################################################################
######################### DEFINING CONTANT VALUES #############################
###############################################################################

from constants import dim, nodes, h_nodes, γ, E_0, extensions

rt2 = sqrt(2)

T_const = 1.5
a_t = .3

# extensions = [['.1sh', '.1exsh', '.1', '.1ex', '.1el', '.1exl'], 
#               ['.dat03', '.dat0329', '.dat', '.she', '.dat29', '.she29']]
user_path = os.getcwd()
OS = platform.system()

###############################################################################
######################## FUNCTION DESCRIPTION SECTION #########################
###############################################################################

##################### EXCTRACTION OF INPUT DATA VALUES ########################
def nuc_definer(z_number):
    """Identifies by charge number the abbreviated name of the element"""
    lib_of_nuclei = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
                     8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al',
                     14: 'Si', 15: 'P',  16: 'S', 17: 'Cl',  18: 'Ar', 19: 'K',
                     20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn',
                     26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga',
                     32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb',
                     38: 'Sr', 39: 'Y',  40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
                     44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
                     50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs',
                     56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm',
                     62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho',
                     68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta',
                     74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
                     80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
                     86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa',
                     92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk',
                     98: 'Cf',  99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 
                     103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh',
                     108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
                     113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts',
                     118: 'Og',  119: 'Uue', 120: 'Ubn', 121: 'Ubu', 122: 'Ubb',
                     123: 'Ubt', 124: 'Ubq', 125: 'Ubp', 126: 'Ubh', 127: 'Ubs',
                     128: 'Ubo', 129: 'Ube', 130: 'Utn', 131: 'Utu'}
    return lib_of_nuclei[int(z_number)]


def odd_axes_elements_adding(matrix, dimensions):
    """Adds array elements along axis related with odd q_i"""
    dim = len(dimensions)
    odd_axes = [i for i, el in enumerate(dimensions) if el % 2 == 0]
    for i in odd_axes:
        odd_part = np.delete(np.flip(matrix, axis=i), -1, axis=i)
        if len(odd_part.shape) > dim:
            mat_neg = np.ones((dim, dim))
            for j in range(dim):
                for k in range(dim):
                    if (k + j) % 2 == 1:
                        mat_neg[j, k] *= -1
            odd_part = odd_part * mat_neg
        matrix = np.concatenate((odd_part, matrix), axis=i)
    return matrix


def tensor_data_reader(line: np.array, dim: int):
    """Additional utilite to reading promts from Fourier data files"""
    mat = np.zeros((4, 4))
    promt = line.copy()
    for i in range(4):
        mat[i, i:] = promt[:4-i]
        mat[i+1:, i] = promt[1:4-i]
        promt = promt[4-i:]
    return mat[-dim:, -dim:]


def fourier_file_data(data_file: str):
    """Extract input data calculated within framework of Fourier nuclear shape
       parametrization consisted all parameters, i.e transport coefficients"""
    exact_place = os.getcwd() # finds exact directory
    if OS == 'Windows':
        os.chdir(path + '\\Fourier shape data\\')
    elif OS == 'Darwin':
        os.chdir(path + '/Fourier shape data/')
    else:
        os.chdir(path + 'Fourier shape data//')    
    data = [line for line in open(data_file, 'r').readlines()
            if line[0] not in ('#', '\n') or line[0].isalpha()]  # reading rows
    os.chdir(exact_place)

    qlim_line = np.array([float(el) for el in data[3].split()[1:]])
    dq_line = np.array([float(el) for el in data[4].split()[1:]])
    nq_line = np.array([int(el) for el in data[5].split()[1:]])

    which_q = np.where(nq_line - 1 > 0)[0]
    dim = len(which_q)
    N_q, dq, qlim = nq_line[which_q], dq_line[which_q], qlim_line[which_q]
    qlim = np.stack([qlim, qlim + (N_q - 1) * dq])

    bs = np.empty(N_q)
    bc = np.empty(N_q)
    bk = np.empty(N_q)
    bx = np.empty(N_q)
    bf = np.empty(N_q)
    r12 = np.empty(N_q)
    rn = np.empty(N_q)
    vol = np.empty(N_q)
    c = np.empty(N_q)
    m_0 = np.empty(np.append(N_q, (dim, dim)))
    f_0 = np.empty(np.append(N_q, (dim, dim)))

    data = data[12:]

    for i, el in enumerate(data[::3]):
        dat_line = np.array(list(map(float, el.split())))
        ind = tuple(np.round((dat_line[which_q] - qlim[0]) / dq).astype(int))
        bs[ind], bc[ind], bk[ind] = dat_line[6:9]
        bf[ind], r12[ind], bx[ind] = dat_line[10:13]
        vol[ind], c[ind], rn[ind] = dat_line[15:]
        dat_line = np.array(list(map(float, data[3 * i + 1].split()[:-1])))
        m_0[ind] = tensor_data_reader(dat_line, dim)
        dat_line = np.array(list(map(float, data[3 * i + 2].split()[:-1])))
        f_0[ind] = tensor_data_reader(dat_line, dim)

    c = odd_axes_elements_adding(c, which_q)
    bs = odd_axes_elements_adding(bs, which_q)
    bc = odd_axes_elements_adding(bc, which_q)
    bk = odd_axes_elements_adding(bk, which_q)

    bf = odd_axes_elements_adding(bf, which_q)
    bf[:, :7] = 1 - bf[:, :7]

    bx = odd_axes_elements_adding(bx, which_q)
    r12 = odd_axes_elements_adding(r12, which_q)
    rn = odd_axes_elements_adding(rn, which_q)
    vol = odd_axes_elements_adding(vol, which_q)

    m_0 = odd_axes_elements_adding(m_0, which_q)
    f_0 = odd_axes_elements_adding(f_0, which_q)

    qlim[0][np.array(bs.shape) != N_q] = - qlim[1][np.array(bs.shape) != N_q]
    N_q = np.array(bs.shape)

    return N_q, dq, qlim, m_0, f_0, bs, bc, bk, bf, r12, bx, vol, c, rn


def potential_reader(A, Z, N_q, dq, qlim, file_extension='.1'):
    """Extract input data calculated within framework of Fourier nuclear shape
       parametrization consisted all parameters, i.e transport coefficients"""
    isotope_name = nuc_definer(Z) + '-' + str(int(A))
    isotope_file = isotope_name + file_extension
    exact_place = os.getcwd()   # finds exact directory
    if OS == 'Windows':
        os.chdir(path + '\\PES data\\')
    elif OS == 'Darwin':
        os.chdir(path + '/PES data/')
    else:
        os.chdir(path + 'PES data//')
    if isotope_file not in os.listdir():
        os.chdir(exact_place)
        print('Error! there no file in PES data directory')
        sys.exit()
    data = open(isotope_file, 'r').readlines()   # reading data from file
    os.chdir(exact_place)

    q_dim_data = [i for i in data[2].split() if i[0] in ['q', 'Q']]
    is_4d = len(q_dim_data) == len(N_q) # check 4D or 3D case

    eld_idx = data[2].split().index('Eld') - 1
    e_tot_idx = data[2].split().index('Etot') - 1
    data = data[3:]
    q_idx = np.arange(4) if is_4d else np.arange(1, 4)
    n_q_file = np.array([N_q[i] // 2 + 1 if el % 2 == 0 else N_q[i] for i,
                         el in enumerate(q_idx)])
    qlim_file = np.array([0 if el % 2 == 0 else qlim[0, i] for i, el in
                          enumerate(q_idx)])
    V_macro = np.empty(n_q_file)
    V_micro = np.empty(n_q_file)

    for i, el in enumerate(data[:n_q_file.prod()]):
        dat_line = np.array(list(map(float, el.split())))
        idx = tuple(np.round((dat_line[q_idx] - qlim_file) / dq).astype(int))
        V_macro[idx] = dat_line[eld_idx]
        V_micro[idx] = dat_line[e_tot_idx] - V_macro[idx]

    V_macro = odd_axes_elements_adding(V_macro, q_idx)
    V_micro = odd_axes_elements_adding(V_micro, q_idx)

    return V_macro, V_micro
###############################################################################


############################ INItIAL POINT CHECK ##############################
def st_pnt_def(a_nuc, z_nuc, saddle_type: str = '2sad'):
    """Extraction of saddle point from database sad_pnt_crds.xlsx"""
    if 'sad_pnt_crds.xlsx' not in os.listdir():
        print('There no library file on main folder.' +
              ' The program will be aborted')
        return sys.exit()
    data = pd.read_excel('sad_pnt_crds.xlsx', sheet_name='Z'+str(int(z_nuc)),
                         engine='openpyxl')
    list_of_q = ['q' + str(i) for i in range(5 - len(dq), 5)]
    type_of_point = '_2_min' if saddle_type == ' 2min' else '_2_sad'
    list_of_q = [ i + type_of_point for i in list_of_q]
    list_of_q.insert(0, 'A')
    data = data.loc[:, list_of_q].to_numpy()
    crd = data[np.where(a_nuc == data[:, 0]), 1:].flatten()
    if np.size(crd) == 0:
        print('There no information about this isotope.' +
              ' The program will be aborted')
        return sys.exit()
    else:
        return crd


def spontaneus_st_point(V, gs, q_2sad_st):
    """Defining initial point in case of spontaneus fission"""
    if q_grid[0][q_2sad_st[0]] < 0.65:
        q_2sad_st[0] = int(round((0.65 - q_grid[0][0]) / dq[0]))
    delta_eps = [.01, .05, 0.1]
    bot, up = N_q[1] // 2  - 5, N_q[1] // 2 + 5
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


def st_pnt_checking(st_point, V:np.array):
    """Checks starting point position"""
    st_idx = np.array([round((el - qlim[0, i]) / dq[i])
                          for i, el in enumerate(st_point)]).astype(int)
    pretendents = np.array(np.where(V == V[tuple(st_idx)].min())).T\
        if len(st_idx) != 1 else np.array(np.where(V == V[st_idx[0],
                                                          2:-3].min())
                                             )
    pretendents = pretendents[pretendents[:, 0] == st_idx[0]]
    pretendents = pretendents[-1] if len(pretendents.shape) > 1 else pretendents
    if (pretendents[:len(st_idx)] != st_idx).any() or len(pretendents) > len(st_idx):
        st_idx = pretendents.copy()
        st_point = st_idx * dq + qlim[0]
    return st_point, st_idx, V[tuple(st_idx)]
###############################################################################


###############################################################################
@nb.njit(fastmath=True, nogil=True)
def density(A, Z, bs:np.array, bk:np.array, bc:np.array)->np.array:
    """Defines multidimentional grid values of density energy function defined
       by Nerlo-Pomorska PRC 2006 paper"""
    result = np.empty_like(bs)
    for idx, el in np.ndenumerate(bs):
        result[idx] = .092 * A + .036 * A**(2 / 3) * el\
                      + .275 * A**(1 / 3) * bk[idx]\
                      - .00146 * Z**2 / A**(1 / 3) * bc[idx]
    return result
###############################################################################


###################### GAUSS-HERMIT SUBROUINES SECTION ########################
@nb.njit(nb.float64[:](nb.float64[:], nb.int64))
def round_njt(x, decimals):
    """JITted version of np.round function"""
    out = np.empty(x.shape[0])
    return np.round_(x, decimals, out)


@nb.njit(fastmath=True)
def gh_ap3d(q, qlim, dq, N_q, matrix):
    """Calculate derivative by G-H method on 3d mesh"""
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(nb.intp)
    f = np.zeros((dim, nodes))
    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2**2) # (1.5 - u ** 2)
        f[i] /= sum(f[i])
    element = 0.
    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1,  crd_int[1] + j - h_nodes))
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                element += f[0, i] * f[1, j] * f[2, k] * matrix[ii, jj, kk]
    return element


@nb.njit(fastmath=True)
def gh_ap3d_tens(q, qlim, dq, N_q, matrix):
    """GH approximation procedure for tensor case on 3d mesh."""
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(np.intp)
    f = np.zeros((dim, nodes))
    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2**2) # (1.5 - u ** 2)
        f[i] /= sum(f[i])

    tens = np.zeros((dim, dim))

    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1,  crd_int[1] + j - h_nodes))
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                tens += f[0, i] * f[1, j] * f[2, k] * matrix[ii, jj, kk]
    return tens


@nb.njit(fastmath=True)
def gh_ap3d_set_without_dq(q, qlim, dq, N_q, ar_invM, ar_G, ar_sqrtG, ar_Vmac,
                           ar_Vmic, ar_den, ar_d_invM, ar_d_Vmac, ar_d_Vmic,
                           ar_d_den):
    """Calculate using by GH method set of needed values."""
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(np.intp)
    f = np.zeros((dim, nodes))
    for i in range(dim):
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = exp(-u2) * (1.875 - 2.5 * u2 + .5 * u2**2) # (1.5 - u ** 2)
        f[i] /= sum(f[i])
    
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
            jj = max(0, min(N_q[1] - 1,  crd_int[1] + j - h_nodes))
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

    return t_invM, t_G, t_rootG, el_Vmac, el_Vmic, el_den, d_el_den, t_dinvM,\
           v_dVmac, v_dVmic

###############################################################################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def shell_correction(T, shell_flag, T_const, a_t):
    return 1 / (1 + exp((T - T_const) / a_t)) if shell_flag else 1.
    # return 1 / (1 + exp((.9 / T - .75) / .09)) if shell_flag else 1.


@nb.njit(fastmath=True, nogil=True)
def friction_temp_correction(T, temp_flag):
    return .7 / (1 + exp((.7 - T) / .25)) if temp_flag else 1.

###############################################################################

@nb.njit(fastmath=True, nogil=True)
def ampl_definer_jit(q_start, d2V_dq2):
    ampl = np.array([E_0 / gh_ap3d(q_start, qlim, dq, N_q, d2V_dq2[i])
                     for i in range(dim)])
    ampl[ampl < 0] *= -1
    ampl = np.sqrt(ampl)
    return ampl


@nb.njit(fastmath=True, nogil=True)
def rand_function(dimentions: int):
    return ξ(0, rt2, dimentions) - ξ(0, rt2, dimentions)


@nb.njit(fastmath=True, nogil=True)
def exit_condition(q, r_neck):
    if abs(gh_ap3d(q, qlim, dq, N_q, vol) - 1) >= 1e-3:
        return True
    return gh_ap3d(q, qlim, dq, N_q, rn) <= r_neck


@nb.njit(fastmath=True, nogil=True)
def q1_def(q_start, V_st, inv_m, ampl, ql):
    E_kin = -1
    if q_start[0] > 0.5:
        while E_kin < 0:
            ξ1 = ξ(0, .5, dim); ξ1[0] = abs(ξ1[0])
            q = q_start + ξ1 * ampl
            if np.any(ql[0] > q) or np.any(ql[1] < q):
                continue
            E_kin = gh_ap3d(q, qlim, dq, N_q, V_st)
    else: 
        while E_kin < 0:
            ξ1 = ξ(0, .5, dim); ξ1[0] = abs(ξ1[0])
            q = q_start + ampl * ξ1
            E_kin = E_0 + (ground_state - gh_ap3d(q, qlim, dq, N_q, V))
    mass = 0.5 * gh_ap3d_tens(q, qlim, dq, N_q, inv_m)
    p_ampl = np.sqrt(E_kin / np.diag(mass)) 
    p = p_ampl * ξ(0, 0.5, dim)
    p *= sqrt(mass @ p @ p / E_kin)
    p[0] = abs(p[0])
    return q, p


@nb.njit(fastmath=True, nogil=True)
def temp_def(temp, p, E_total, i_m, a, V_mac, V_mic, shell):
    E_kin = .5 * i_m @ p @ p 
    E_st = E_total - (E_kin + V_mac + V_mic * shell)
    if E_st < 0:
        E_kin_new = E_kin + E_st - a * temp ** 2 
        p *= sqrt(E_kin_new / E_kin) if E_kin_new > 0 else 0
    temp2 = max(E_st / a, 1e-16)
    temp = max(temp, sqrt(temp2))
    t_star = sqrt(E_0 / tanh(E_0 / temp)) if t_star_enable else sqrt(temp)
    shell = shell_correction(temp, shell_ef, T_const, a_t)
    g_coef = friction_temp_correction(temp, temp_ef)
    return temp, temp2, t_star, shell, g_coef, E_st, p

###############################################################################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def trajectory_calc(q_start, temperature, inv_m, fric, sqrt_fric, V_macro,
                    V_micro, V_s2, a_d, r_neck, sigma_r_neck, temp_ef,
                    shell_ef, t_star_enable, dt, sqrt_dt, idt, step_limit,
                    ql, ampl):
    """Caclulates trajectory of Monte Carlo processes"""
    dim = len(q_start)
    temp = temperature
    time = 0
    shell = shell_correction(temp, shell_ef, T_const, a_t)
    dp = np.zeros(dim)
    q2_max = qlim[1, 0]

    q, p = q1_def(q_start, V_s2, inv_m, ampl, ql)

    all_data_pack = (qlim, dq, N_q, inv_m, fric, sqrt_fric, V_macro,
                     V_micro, a_d, d_i_m_dq, dV_macro_dq, dV_micro_dq,
                     d_a_d_dq)

    r_neck_traj = abs(ξ(r_neck, sigma_r_neck)) if gauss_flag else r_neck
    
    while q_start[0] <= q[0]:

        i_m, g, sqrt_g, V_mac, V_mic, a, d_a, d_i_m,\
             dV_mac, dV_mic = gh_ap3d_set_without_dq(q, *all_data_pack)

        if time % idt == 0:
            temp, temp2, t_star, shell, g_coef, E_st, p =\
                temp_def(temp, p, E_total, i_m, a, V_mac, V_mic, shell)
            if poisson_flag:
                r_neck_traj = np.random.poisson(temp / 5) * r_nucleon
        g *= g_coef
        sqrt_g *= sqrt(g_coef)

        dp = np.array([sqrt_g[i].dot(rand_function(dim)) * sqrt_dt * t_star
                       - (.5 * d_i_m[i] @ p + i_m @ g[i]) @ p * dt
                       for i in range(dim)])
        dp -= dt * (dV_mac + shell * dV_mic - d_a * temp2)
        p += dp
        Δq = i_m @ (p - dp / 2)
        q += Δq * dt

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
            return q, p, time * dt, temp_def(temp, p, E_total, i_m, a, V_mac,
                                             V_mic, shell)[0], q[2] < 0
        elif exit_condition(q, r_neck_traj) and q[0] > 1.5:
            temp = temp_def(temp, p, E_total, i_m, a, V_mac, V_mic, shell)[0]
            return q, p, time * dt, temp, temperature + 0.1 < temp

    return q, p, time * dt, temp_def(temp, p, E_total, i_m, a, V_mac,
                                     V_mic, shell)[0], False


@nb.njit(fastmath=True, nogil=True, parallel=True)
def multithreading_trj_calc(input_param, N_tr):
    
    wrong_count = 0
    traj_crd_out = np.empty((N_tr, len(input_param[0])))
    traj_p_out = np.empty_like(traj_crd_out)
    traj_time = np.empty(N_tr)
    traj_temp = np.empty(N_tr)
    for i in nb.prange(N_tr):
        correct_traj = False
        while not correct_traj:
            q, p, t, temp_out, correct_traj = trajectory_calc(*input_param)
            wrong_count += 0 if correct_traj else 1
        traj_crd_out[i] = q.copy()
        traj_p_out[i] = p.copy()
        traj_time[i] = t
        traj_temp[i] = temp_out
    print(f' \tTotal:\t   {N + wrong_count}\n \tNot passed: {wrong_count}\n')
    return traj_crd_out, traj_p_out, traj_time, traj_temp


def monte_carlo(q_start, temperature, d2V_dq2, inv_m, fric, sqrt_fric, V_macro,
                V_micro, V, a_d, r_neck, sigma_r_neck,
                dt:float=.01, N:int=3000, T_const:float=1.5, a_t:float=.3):
    """Calculates set of N trajectories with random starting point"""

    ampl = ampl_definer_jit(q_start, d2V_dq2)
    ql = np.array([q_start - np.array([0, 3, 3]) * dq,
                   q_start + np.array([4, 3, 3]) * dq])
    ql[0][ql[0] < qlim[0]] = qlim[0][ql[0] < qlim[0]]
    ql[1][ql[1] > qlim[1]] = qlim[1][ql[1] > qlim[1]]
    V_st = V_starting - V - E_0
    
    inpt = (q_start, temperature, inv_m, fric, sqrt_fric, V_macro, V_micro,
            V_st, a_d, r_neck, sigma_r_neck, temp_ef, shell_ef, t_star_enable,
            dt, sqrt(dt), int(round(.1 / dt)), int(100000 / dt), ql, ampl)
    
    trj_q_out, trj_p_out, trj_time, trj_T = multithreading_trj_calc(inpt, N)
    
    return trj_q_out, trj_p_out, trj_time, trj_T 

###############################################################################
###############################################################################

def r_fit_procedure(q_start, temperature, d2V_dq2, inv_m, fric, sqrt_fric,
                    V_macro, V_micro, V, a_d, dt: float = .01,
                    N: int = 3000, T_const: float = 1.5, a_t: float = .3):
    r_n_var = np.linspace(.1, .4, 13)
    a_r_n_var = np.linspace(.01, .1, 10)

    var_grid = np.meshgrid(r_n_var, a_r_n_var)
    I = np.empty(var_grid[0].shape)
    out = [[[],] * len(r_n_var) for _ in a_r_n_var]
    exp_dist = exp_res_aut(exp_file, path)
    p_bar = tqdm(total=np.prod(I.shape), position=0, leave=True)
    for idx, (r_n, a_r_n) in zip(np.ndindex(I.shape), np.nditer(var_grid)):
        r, ar = float(r_n), float(a_r_n)
        q_out, p_out, traj_time, temp_out = monte_carlo(q_start, temperature,
                                                        inv_m, fric, sqrt_fric,
                                                        V_macro, V_micro, V, 
                                                        a_d, r, ar, dt, N)
        A_h_0 = np.round(.5 * A * (1 + aux.q_into_alpha(q_out)))
        A_h = np.array([i_e for i_e in A_h_0 if 0 <= i_e <= A])
        I[idx] = dist_comparision(A_h, exp_dist)
        out[idx[0]][idx[1]] = (q_out.copy(), p_out.copy(),
                               traj_time.copy(), temp_out.copy()
                              )
        p_bar.update(1)
    p_bar.close(); print()

    loc = tuple([o[0] for o in np.where(I == I.min())])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(r_n_var, a_r_n_var, I)
    plt.colorbar(ax.contourf(r_n_var, a_r_n_var, I))
    ax.set_xlabel('$R_{neck}$, $R_0$', size=30)
    ax.set_ylabel(r'$a_{R_n}$, $R_0$', size=30)
    plt.title(nuc_definer(Z) + '-' + f'{int(A)}')
    fig.tight_layout()
    plt.tick_params(axis='both', which='major', direction='inout',
                    labelsize='large', bottom=True, top=True, left=True,
                    right=True, labelbottom=True, labelleft=True)
    return out[loc[0]][loc[1]], r_n_var[loc[1]], a_r_n_var[loc[0]],\
            (I, I.min())


def r_fit_procedure_mod(q_start, temperature, d2V_dq2, inv_m, fric, sqrt_fric,
                        V_macro, V_micro, V, a_d, dt: float = .01,
                        N: int = 3000, T_const: float = 1.5, a_t: float = .3):

    r_n_var = np.linspace(.5, 3, 6) * r_nucleon
    a_r_n_var = np.linspace(.25, 1.25, 5) * r_nucleon

    var_grid = np.meshgrid(r_n_var, a_r_n_var)
    I = np.empty(var_grid[0].shape)
    out = [[[],] * len(r_n_var) for _ in a_r_n_var]
    exp_dist = exp_res_aut(exp_file, path)
    pbar = tqdm(total=np.prod(I.shape), position=0, leave=True)
    for idx, (r_n, a_r_n) in zip(np.ndindex(I.shape), np.nditer(var_grid)):
        
        r, ar = float(r_n), float(a_r_n)
        q_out, p_out, traj_time, temp_out = monte_carlo(q_start, temperature,
                                                        d2V_dq2, inv_m, fric,
                                                            sqrt_fric, V_macro,
                                                            V_micro, V, a_d, r,
                                                            ar, dt, N)

        A_h_0 = np.round(.5 * A * (1 + aux.q_into_alpha(q_out)))
        A_h = np.array([i_e for i_e in A_h_0 if 0 <= i_e <= A])
        I[idx] = dist_comparision(A_h, exp_dist)
        out[idx[0]][idx[1]] = (q_out.copy(), p_out.copy(),
                               traj_time.copy(), temp_out.copy()
                              )
        pbar.update(1)
    pbar.close(); print()

    loc = tuple([o[0] for o in np.where(I == I.min())])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(r_n_var, a_r_n_var, I)
    plt.colorbar(ax.contourf(r_n_var, a_r_n_var, I))
    ax.set_xlabel('$R_{neck}$, $R_0$', size=30)
    ax.set_ylabel(r'$a_{R_n}$, $R_0$', size=30)
    plt.title(nuc_definer(Z) + '-' + f'{int(A)}')
    fig.tight_layout()
    plt.tick_params(axis='both', which='major', direction='inout',
                    labelsize='large', bottom=True, top=True, left=True,
                    right=True, labelbottom=True, labelleft=True)
    
    return out[loc[0]][loc[1]], r_n_var[loc[1]], a_r_n_var[loc[0]], (I, I.min())


def exp_res_aut(file_name, dir_path):
    exact_place = os.getcwd() # finds exact directory
    if OS == 'Windows':
        os.chdir(dir_path + 'Experimental data\\')
    elif OS == 'Darwin':
        os.chdir(dir_path + '/Experimental data/')
    else:
        os.chdir(dir_path + 'Experimental data//')
    A = ''.join([num for num in file_name[:5] if num.isdigit()])
    A = int(A)
    file = open(file_name, 'r')
    text = file.readlines()
    file.close()
    os.chdir(exact_place)

    data = [line for line in text if line[0] not in ('\n', '#')
            and not line[0].isalpha()]
    nucl_yield = []
    nucl_bins = []
    for line in data:
        line_list = list(map(float, line.split()))
        if 0 <= line_list[0] <= A:
            nucl_bins.append(line_list[0])
            nucl_yield.append(line_list[1])
    if sum(nucl_yield) > 3:
        nucl_yield = [i / 100 for i in nucl_yield]
    if nucl_bins[0] >= int(A // 2) or nucl_bins[-1] <= int(A // 2) + 1:
        nucl_bins1 = [A - i for i in reversed(nucl_bins[:-1])]
        nucl_yield1 = nucl_yield[1:].copy()
        nucl_yield1.reverse()
        nucl_bins = nucl_bins1 + nucl_bins
        nucl_yield = nucl_yield1 + nucl_yield
    return np.array(nucl_bins), np.array(nucl_yield)


def dist_comparision(A_h, exp_dist):
    exp_bins, exp_yld = exp_dist
    exp_bins = np.round(exp_bins)
    yld, bins = np.histogram(A_h, bins=int(max(A_h) - min(A_h)), density=True)
    bins = bins[:-1]
    (long_ar, short_ar) = (bins.copy(), exp_bins.copy()) if len(bins) >= len(exp_bins)\
        else (exp_bins.copy(), bins.copy())
    yld_ex = np.empty_like(short_ar)
    exp_yld_ex = np.empty_like(short_ar)
    for i, el in enumerate(short_ar):
        yld_ex[i] = yld[bins == el] if yld[bins == el].size == 1 else 0
        exp_yld_ex[i] = exp_yld[exp_bins == el]\
            if exp_yld[exp_bins == el].size == 1 else 0
    return sum(np.abs(exp_yld_ex - yld_ex))


def fit_procedure(A, exp_file, dir_path):
    t_c = np.linspace(1.4, 2, 13)
    a_t = np.linspace(.2, .4, 11)
    var_grid = np.meshgrid(t_c, a_t)
    I = np.empty(var_grid[0].shape)
    exp_dist = exp_res_aut(exp_file, dir_path)
    pbar = tqdm(total=np.prod(I.shape), position=0, leave=True)
    for idx, (T_cons_var, a_t_var) in zip(np.ndindex(I.shape),
                                          np.nditer(var_grid)):
        T = sqrt(E_star / gh_ap3d(starting_point, qlim, dq, N_q, a_d))
        sh = shell_correction(T, shell_ef, T_cons_var, a_t_var)
        F = V_macro + sh *  V_micro - a_d * T ** 2
        dF_dq = np.array(np.gradient(F, dq[0], dq[1], dq[2]))
        d2F_dq2 = np.array([np.gradient(el, dq[0], dq[1], dq[2])[i]
                            for i, el in enumerate(dF_dq)])
        dF_d2F_vec = np.array([el[tuple(start_idx)] / d2F_dq2[i][tuple(start_idx)]
                     if i != 0 else 0 for i, el in enumerate(dF_dq)])
        q_out = monte_carlo(starting_point, T, d2F_dq2, r_neck, sigma_r_neck,
                            dt, N, T_const=T_cons_var, a_t=a_t_var)[0]
        A_h_0 = np.round(.5 * A * (1 + aux.q_into_alpha(q_out)))
        A_h = np.array([i_e for i_e in A_h_0 if 0 <= i_e <= A])
        I[idx] = dist_comparision(A_h, exp_dist)
        pbar.update(1)
    pbar.close(); print()

    loc = np.where(I == I.min())

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(t_c, a_t, I)
    plt.colorbar(ax.contourf(t_c, a_t, I))
    ax.set_xlabel('$T_{coef}$, MeV', size=30)
    ax.set_ylabel(r'$a_{T}$, MeV', size=30)
    fig.tight_layout()
    plt.tick_params(axis='both', which='major', direction='inout',
                    labelsize='large', bottom=True, top=True, left=True,
                    right=True, labelbottom=True, labelleft=True)
    return I, I.min(), (t_c[loc[1]], a_t[loc[0]])

###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":

    if 'input.xlsx' not in os.listdir():
        print('Error: there no input file!')
        sys.exit()
    print('Calculations starts: ' +
          datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S"))

    Z_prev, A_prev = 0, 0
    fourier_file_prev = ''
    prev_pot_file_ext = ''
    inp_data = pd.read_excel('input.xlsx')

    exact_place = os.getcwd()
    path = os.getcwd()

    num_ptn = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(num_ptn, re.VERBOSE)

    for i, isotope in inp_data.iterrows():
        if isotope.isnull().all():
            sys.exit()
        Z, A, N, E_init, dt, starting_point, temp_ef, shell_ef,\
            t_star_enable, r_neck, sigma_r_neck, diffiuse_mult,\
                exp_file, fit_of_T, gauss_flag, poisson_flag, elong_flag,\
                    short_q2_flg, limit_cut_flag, info_full = isotope

        gauss_flag = False\
            if type(gauss_flag) != bool else gauss_flag and (not isnan(r_neck))
        # poisson_flag = False\
            # if type(poisson_flag) != bool and isnan(r_neck) else poisson_flag
        poisson_flag = poisson_flag and isnan(r_neck)
        elong_flag = False if type(elong_flag) != bool else elong_flag
        limit_cut_flag = False if type(limit_cut_flag) != bool else limit_cut_flag
        short_q2_flg = False if type(short_q2_flg) != bool else short_q2_flg

        if i != 0:
            for func in [ampl_definer_jit, trajectory_calc, gh_ap3d, gh_ap3d_tens]:
                func.recompile()

        diffiuse_mult = 1 if isnan(diffiuse_mult) else sqrt(diffiuse_mult)
        if diffiuse_mult > 1:
            rt2 = diffiuse_mult
        Z, A, N = int(Z), int(A), int(N)

        SHE_flag = Z > 103

        fl_m = np.array([not(SHE_flag or elong_flag) and short_q2_flg,
                         (not SHE_flag) and elong_flag and short_q2_flg,
                         not(SHE_flag or elong_flag) and (not short_q2_flg),
                         SHE_flag and not elong_flag and (not short_q2_flg),
                         not SHE_flag and elong_flag and (not short_q2_flg),
                         SHE_flag and elong_flag and (not short_q2_flg)
                         ]
                        )
        pot_file_ext  = np.array(extensions[0])[fl_m][0]
        fourier_file = 'fourier' + np.array(extensions[1])[fl_m][0]

        if fourier_file_prev != fourier_file:
            N_q, dq, qlim, m_0, f_0, bs, bc, bk, bf,\
                r12, bx, vol, c, rn = fourier_file_data(fourier_file)
            fourier_file_prev = fourier_file
            dim = len(dq)
            q_grid = [np.linspace(qlim[0, i], qlim[1, i], N_q[i])
                      for i in range(dim)]

        if (Z, A) != (Z_prev, A_prev):
            r0 = 1.2 * A ** (1/3)
            r_nucleon = 1 / r0
            m_cf = 0.0113 * A ** (5 / 3)
            fric_cf = 0.275 * A ** (4 / 3)
            m = m_0.copy() * m_cf
            fric = f_0.copy() * fric_cf
            inv_m = np.zeros_like(m)
            sqrt_fric = np.zeros_like(fric)
            for idx, el in np.ndenumerate(bs):
                sqrt_fric[idx] = lalg.sqrtm(fric[idx])
                inv_m[idx] = lalg.inv(m[idx])

            d_i_m_dq = np.zeros(tuple([dim,] + [i for i in inv_m.shape]))
            for i in range(dim):
                for j in range(dim):
                    d_i_m_dq[..., i, j] = np.array(np.gradient(inv_m[..., i, j],
                                                               dq[0], dq[1],
                                                               dq[2],
                                                               edge_order=2))

            a_d = density(A, Z, bs, bk, bc)  # aden.copy()
            Z_prev, A_prev = Z, A

        if pot_file_ext != prev_pot_file_ext:
            prev_pot_file_ext = pot_file_ext
            V_macro, V_micro = potential_reader(A, Z, N_q, dq, qlim,
                                                pot_file_ext)
            V = V_macro + V_micro

            d_a_d_dq = np.array(np.gradient(a_d, dq[0], dq[1], dq[2]))
            dV_macro_dq = np.array(np.gradient(V_macro, dq[0], dq[1], dq[2]))
            dV_micro_dq = np.array(np.gradient(V_micro, dq[0], dq[1], dq[2]))

            q_gs_lim = np.array([[.3, -.15, -.15],
                                 [.6, .15, .15]]) \
                       if short_q2_flg else \
                       np.array([[0, -.15, -.15],
                                 [.5, .15, .15]])

            area_idx = ((q_gs_lim - qlim[0]) / dq).T.astype(int)
            gs_area = np.array([slice(i[0], i[1]) for i in area_idx])
            ground_state = V[tuple(gs_area)].min()
            gs_mesh_crd = tuple(map(int,
                                    np.array(np.where(V == ground_state)).T[0]))
            q2_gs = q_grid[0][gs_mesh_crd[0]]

        if type(starting_point) == str:
            is_number = rx.findall(''.join(starting_point))
            if 'from file' in starting_point:
                starting_point = st_pnt_def(A, Z,
                                            starting_point.split('from file')[-1])
                starting_point, start_idx,\
                    V_starting = st_pnt_checking(starting_point, V)
            elif starting_point in ['spont', 'spontaneus']:
                q_2sad_idx = ((st_pnt_def(A, Z) - qlim[0]) / dq).astype(int)
                starting_point, start_idx,\
                    V_starting = spontaneus_st_point(V, ground_state, q_2sad_idx)
            else:
                starting_point = np.array(list(map(float, is_number)))
                starting_point, start_idx,\
                    V_starting = st_pnt_checking(starting_point, V)
        else:
            print('Error! Wrong starting point input')

        E_star = E_init 
        E_total = E_init + st_pnt_checking(st_pnt_def(A, Z), V)[-1]\
                    - ground_state

        if E_init < 0:
            print('Error! Invalid initial energy value.' +
                  ' Please check input parameteres.')
            sys.exit()

        if not isnan(fit_of_T):
            os.chdir(path + 'Experimental data\\')
            if exp_file in os.listdir():
                fit = fit_procedure(A, exp_file, exact_place)
                T_const, a_t = fit[2]
            else:
                print('There no file ' + exp_file + ' in experimental data' +
                      ' directory')
            os.chdir(exact_place)
        temperature = sqrt(E_star / gh_ap3d(starting_point, qlim,
                                            dq, N_q, a_d)
                           )
    
        sh = shell_correction(temperature, shell_ef, T_const, a_t)
        F = V_macro + sh *  V_micro - a_d * temperature ** 2

        dV_dq = np.array(np.gradient(V, dq[0], dq[1], dq[2]))
        d2V_dq2 = np.array([np.gradient(el, dq[0], dq[1], dq[2])[i]
                            for i, el in enumerate(dV_dq)])

        isotope_name = nuc_definer(Z) + '-' + f'{int(A)}'
        print('\t Isotope ' + isotope_name + f' E = {round(E_star, 3)}')

        inp_var = starting_point, temperature, d2V_dq2, inv_m, fric,\
                    sqrt_fric, V_macro, V_micro, V, a_d
        if isnan(r_neck) and isnan(sigma_r_neck) and type(exp_file) == str:
            (q_out, p_out, traj_time, temp_out), r_neck, sigma_r_neck,\
                fit_out = r_fit_procedure_mod(*inp_var, dt, N)
                # fit_out = r_fit_procedure(*inp_var, dt, N)
        else:
            q_out, p_out, traj_time,\
                  temp_out = monte_carlo(*inp_var, r_neck,
                                         sigma_r_neck, dt, N)

        rn_out = np.array([gh_ap3d(i, qlim, dq, N_q, rn) for i in q_out])
        output = pd.DataFrame({'time': traj_time,
                               'q2': q_out[:, 0], 'q3': q_out[:, 1],
                               'q4': q_out[:, 2],
                               'p2': p_out[:, 0], 'p3': p_out[:, 1],
                               'p4': p_out[:, 2],
                               'Rneck': rn_out, 'Temperature': temp_out})

        res_path = path + '\\Result\\' if OS == 'Windows' else path + '/Result/'
        if not os.path.isdir(res_path):
            os.mkdir(res_path)
        res_path += f'{datetime.datetime.today().strftime("%d-%m-%y")}'
        if not os.path.isdir(res_path):
            os.mkdir(res_path)
        os.chdir(res_path)

        add_p = 'P ' if poisson_flag else ''
        add_g = 'G ' if gauss_flag else ''
        add_lim = 'q2 lim' if limit_cut_flag else 'q2 prob'
        e0 = 'e0_{}'.format(E_0).replace('.', '')

        I_fit = ''
        if 'fit_out' in globals():
            pd.DataFrame(fit_out[0]).to_csv(isotope_name + ' I(R, σ) table.csv')
            I_fit = f'(I = {fit_out[1]:.4g} {exp_file})'

        file_name = (f'N = {len(q_out)} dt= {dt} {add_g}{add_p}'
                      + f'R_n= {r_neck * r0:.4g} with σ = {sigma_r_neck:.4g}'
                      + ' at q_st = ' 
                      + f'{" ".join(str(starting_point.round(3)).split())}'[1:-1]
                      + f' E = {round(E_star, 3)} '
                      + f't_cor = {str(temp_ef)} sh_cor = {shell_ef} '
                      + f't_star = {t_star_enable} '
                      + f'& {pot_file_ext[1:]} pot type '
                      + f'at ✓{diffiuse_mult**2:.2g} '
                      + f'{e0} {add_lim}' + I_fit
                      + '.xlsx') # &    q2 abs unlim randint _int_nck choice123 q2max25 q234bnd
        
        output.to_excel(isotope_name + ' ' + file_name, sheet_name='Sheet1',
                        engine='openpyxl', index=False)
        
        emission_FF_flag = True
        
        if info_full in ('p', 'pre', 'precise'):
            param_cf = np.array([aux.q_to_nucl_param(q) for q in q_out])
            # param_cf = np.array([aux.fcs_pythonic(q) for q in q_out])
            Bf_q = param_cf[:, 0]
            A_f_0 = np.round(A * Bf_q).astype(int)
            Z_f_0 = np.round(Z * Bf_q).astype(int)
            R12_q = r0 * param_cf[:, 1]
            BCoul_q, Bs_q, Bc_q = param_cf[:, 2], param_cf[:, 3], param_cf[:, 4]
            if emission_FF_flag:
                emsn_out = [em.emission_FF(A, Z, A_LF, Z_LF, T, el)
                            for A_LF, Z_LF, T, el in zip(A_f_0, Z_f_0,
                                                         temp_out,
                                                         param_cf[:, 1:])
                            ]
                A_f_1, Z_f_1, E_star_CN,\
                    E_star_FF, e_n_FF = em.emFF_decoder(emsn_out)

        elif info_full in ('a','approx', 'approximate'):
            Bf_q = .5 * (1 + aux.q_into_alpha(q_out))
            # Bf_q = np.array({gh_ap3d(q, qlim, dq, N_q, bf) for q in q_out})
            A_f_0 = np.round(A * Bf_q).astype(int)
            Z_f_0 = np.round(Z * Bf_q).astype(int)
            Bs_q = np.array([gh_ap3d(q, qlim, dq, N_q, bs) for q in q_out])
            Bc_q = np.array([gh_ap3d(q, qlim, dq, N_q, bc) for q in q_out])
            BCoul_q = np.array([gh_ap3d(q, qlim, dq, N_q, bc) for q in q_out])
            R12_q = r0 * np.array([gh_ap3d(q, qlim, dq, N_q, r12)
                                   for q in q_out])
            if emission_FF_flag:
                emsn_out = [em.emission_FF_gh3d(q, A, Z, T, qlim, dq, N_q, bc,
                                                bs, bk)
                            for (q, T) in zip(q_out, temp_out)]
                A_f_1, Z_f_1, E_star_CN,\
                    E_star_FF, e_n_FF = em.emFF_decoder(emsn_out)
        else:
            os.chdir(exact_place)
            sys.exit()

        output_2 = pd.DataFrame({'Af_0' : A_f_0,
                                 'Zf_0' : Z_f_0,
                                 'Bf'   : Bf_q,
                                 'Bs'   : Bs_q,
                                 'Bc'   : Bc_q,
                                 'BCoul': BCoul_q,
                                 'R12'  : R12_q})

        A_f = np.concatenate((A_f_0, A - A_f_0))
        Z_f = np.concatenate((Z_f_0, Z - Z_f_0))

        Af_range = np.arange(A_f.min(), A_f.max() + 2, dtype=int)
        Zf_range = np.arange(Z_f.min(), Z_f.max() + 2, dtype=int)

        h, Af_range, Zf_range = np.histogram2d(A_f, Z_f,
                                               bins=(Af_range, Zf_range),
                                               density=True)
        h *= 2
        Z_A = np.empty((1,3))
        for i, el in enumerate(h.T):
            el_mask = ~np.isclose(el, 0)
            if any(el_mask):
                Z_A = np.concatenate((Z_A,
                                      np.array([[Zf_range[i], j, k]
                                      for j,k in zip(Af_range[:-1][el_mask],
                                                     el[el_mask])
                                                ])
                                      ))
        Z_A = Z_A[1:]

        output_3YA = pd.DataFrame({"Af": Af_range[:-1],
                                   "Y(Af)": h.sum(axis=1)
                                   }
                                  )

        output_3YZ = pd.DataFrame({"Zf": Zf_range[:-1],
                                   "Y(Zf)": h.sum(axis=0)
                                   }
                                  )

        output_3_YZA = pd.DataFrame({"Zf": Z_A.T[0].astype(int),
                                     "Af": Z_A.T[1].astype(int),
                                     "Y(Zf,Af)": Z_A.T[2]
                                     }
                                    )

        if emission_FF_flag:

            Af1_range = np.arange(A_f_1.min(), A_f_1.max() + 2, dtype=int)
            Zf1_range = np.arange(Z_f_1.min(), Z_f_1.max() + 2, dtype=int)

            h_1, Af1_range, Zf1_range = np.histogram2d(A_f_1.flatten(),
                                                       Z_f_1.flatten(),
                                                       bins=(Af1_range,
                                                             Zf1_range),
                                                       density=True)
            h_1 *= 2
            Z_A_1 = np.empty((1,3))
            for i, el in enumerate(h_1.T):
                el_mask = ~np.isclose(el, 0)
                if any(el_mask):
                    Z_A_1 = np.concatenate((Z_A_1,
                                          np.array([[Zf1_range[i], j, k]
                                          for j,k in zip(Af1_range[:-1][el_mask],
                                                         el[el_mask])
                                                    ])
                                          ))
            Z_A_1 = Z_A_1[1:]
            
            output_3YA_1 = pd.DataFrame({"A'f": Af1_range[:-1],
                                         "Y(A'f)": h_1.sum(axis=1)
                                         }
                                        )

            output_3YZ_1 = pd.DataFrame({"Z'f": Zf1_range[:-1],
                                         "Y(Z'f)": h_1.sum(axis=0)
                                         }
                                        )

            output_3_YZA_1 = pd.DataFrame({"Z'f": Z_A_1.T[0].astype(int),
                                           "A'f": Z_A_1.T[1].astype(int),
                                           "Y(Z'f,A'f)": Z_A_1.T[2]
                                           }
                                          )


        with pd.ExcelWriter(isotope_name + ' ' + file_name,
                            engine='openpyxl') as wr:
            shift_idx = 3 if emission_FF_flag else 0
            output.to_excel(wr, sheet_name='Sheet1', index=False)
            output_2.to_excel(wr, sheet_name='Sheet2', index=False)
            output_3YA.to_excel(wr, sheet_name='Sheet3', startcol=0,
                                index=False)
            output_3YZ.to_excel(wr, sheet_name='Sheet3',
                                startcol= 2 + shift_idx,
                                index=False)
            output_3_YZA.to_excel(wr, sheet_name='Sheet3',
                                  startcol= 7 + shift_idx,
                                  index=False)
            if emission_FF_flag:
                output_3YA_1.to_excel(wr, sheet_name='Sheet3',
                                      startcol= shift_idx - 1,
                                      index=False)
                output_3YZ_1.to_excel(wr, sheet_name='Sheet3',
                                      startcol= 3 * shift_idx - 2,
                                      index=False)
                output_3_YZA_1.to_excel(wr, sheet_name='Sheet3',
                                        startcol= 4 * shift_idx + 1,
                                        index=False)
            
        os.chdir(exact_place)

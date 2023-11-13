###############################################################################
##################### CALLING USED LIBRARIES & PACKAGES #######################
###############################################################################

import os
import sys
import random
import numpy as np
import pandas as pd
from math import pi, exp, sqrt, isnan, copysign

from scipy import integrate as s
import numba as nb
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from numba.typed import List
from numba.types import float64, int64
import auxiliary_library as aux

###############################################################################
###############################################################################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def B_surf(alpha2: float):
    """
        Calculation of B_surf coefficient within global deformation parameter 
        approach porposed in Nix & Swiatecki paper [Nucl. Phys. 71, 1 (1965)]
    """
    return 1 + .4 * alpha2 - 4 / 105 * alpha2 ** (1.5) if alpha2 != 0 else 1


@nb.njit(fastmath=True, nogil=True)
def B_curv(alpha2: float):
    """
        Calculation of B_curv coefficient within global deformation parameter 
        approach
    """
    return 1 + .4 * alpha2 + 16 / 105 * alpha2 ** (1.5) if alpha2 != 0 else 1


@nb.njit(fastmath=True, nogil=True)
def B_coul(alpha2: float):
    """
        Calculation of B_Coulomb coefficient within global deformation
        parameter approach
    """
    return 1 - .2 * alpha2 - 4 / 105 * alpha2 ** (1.5) if alpha2 != 0 else 1


@nb.njit(fastmath=True, nogil=True)
def LSD(A, Z, alpha2 = 0):
    """
        Calculates the mass of nuclei using Lublin-Strasbourg formula taken
        from Pomorski & Dudek paper [PRC  67, 044316 (2003)].
        Deformation coefficient defined via Nix & Swiatecki GDP approach
    """
    A13 = A ** (1/3)
    A, Z, N = np.array([A, Z, A - Z]).astype(nb.int16)
    I = (N - Z) / A
    # L_term = 36.2865 / A ** (5 / 3) * l * (l + 1) * Bx(alpha2)
    z_terms = (0.70978 / A13 * B_coul(alpha2) - 1.433e-5 * Z ** (0.39)
               - .9181 / A) * Z ** 2
    curv_terms = (16.9707 * (1 - 2.2938 * I ** 2) * A13 ** 2 * B_surf(alpha2)
                  + 3.8602 * (1 + 2.3764 * I ** 2) * A13 * B_curv(alpha2)
                  - 15.492 * (1 - 1.8601 * I ** 2) * A)
    return ΔM_p * Z + ΔM_n * N + z_terms + curv_terms\
            - 10 * exp(- 4.2 * abs(I)) + odd_term(A, Z)
    # return z_terms + curv_terms - 10 * exp(- 4.2 * abs(I)) + odd_term(A, Z)


@nb.njit(fastmath=True, nogil=True)
def odd_term(A, Z):
    """
        Calculates the even-odd term in LSD-formula
    """
    N = A - Z
    n_odd_fl, z_odd_fl, eq_fl =  N % 2, Z % 2, Z == N        
    if eq_fl:
        return (4.8 / Z ** (1/3) + 4.8 / N ** (1/3)
                 - 6.6 / A ** (2/3) + 30 / A) if n_odd_fl else 0
    else:
        if n_odd_fl:
            return (4.8 / Z ** (1/3) + 4.8 / N ** (1/3)
                     - 6.6 / A ** (2/3)) if z_odd_fl else 4.8 / N ** (1/3)
        return 4.8 / Z ** (1/3) if z_odd_fl else 0

    # if z_odd_fl and eq_fl:
    #     return 4.8 / Z ** (1/3) + 4.8 / N ** (1/3) - 6.6 / A ** (2/3) + 30 / A
    # if n_odd_fl and z_odd_fl:
    #     return 4.8 / Z ** (1/3) + 4.8 / N ** (1/3) - 6.6 / A ** (2/3)
    # if z_odd_fl:
    #     return 4.8 / Z ** (1/3)
    # elif n_odd_fl:
    #     return 4.8 / N ** (1/3)
    # return 0


@nb.njit(fastmath=True, nogil=True)
def LSD_def(A, Z, B_coul, B_surf, B_curv):
    A13 = A ** (1/3)
    N = A - Z
    I = (N - Z) / A
    z_terms = (0.70978 / A13 * B_coul - 1.433e-5 * Z ** (0.39)
               - .9181 / A) * Z ** 2
    curv_terms = (16.9707 * (1 - 2.2938 * I ** 2) * A13 ** 2 * B_surf
                  + 3.8602 * (1 + 2.3764 * I ** 2) * A13 * B_curv
                  - 15.492 * (1 - 1.8601 * I ** 2) * A)
    return Z * ΔM_p + N * ΔM_n + z_terms + curv_terms\
            - 10 * exp(- 4.2 * abs(I)) + odd_term(A, Z)


@nb.njit(fastmath=True, nogil=True)
def m_ex_np_ver(A, Z, np_m_data):
    """
        Searching mass excess from AME data table in case of absense
        calculates by LSD 
    """
    A, Z =int(A), int(Z)
    search = np_m_data[np_m_data[:, 0] == A]
    search = search[search[:, 1] == Z]
    if search.size == 0:
        return LSD(A, Z, 0)
    return search[0, -1]


@nb.njit(fastmath=True, nogil=True)
def lvl_dens_param(A, Z):
    I2 = (1 - 2 * Z / A) ** 2
    return .0126 * (1 - 6.275 * I2) * A + .3804 * (1 - 1.101 * I2) * A**(2 / 3)\
            + 1.4e-4 * Z**2 / A**(1 / 3)


@nb.njit(fastmath=True, nogil=True)
def density_deformed(A, Z, B_c, B_s, B_cur):
    """Defines density energy function from Nerlo-Pomorska paper"""
    return .092 * A + .036 * A**(2 / 3) * B_s + .275 * A**(1 / 3) * B_cur\
           - .00146 * Z**2 / A**(1 / 3) * B_c


@nb.njit(fastmath=True)
def den_lvl_float(enrg, A, Z):
    f = lvl_dens_param(A, Z) * enrg
    return sqrt_pi / (12 * enrg * f ** 0.25) * exp(2 * sqrt(f)) if f > 0\
        else np.inf


@nb.njit(fastmath=True)
def den_lvl_arr(e, A, Z):
    f = lvl_dens_param(A, Z) * e
    return sqrt_pi / (12 * e * f ** 0.25) * np.exp(2 * np.sqrt(f))


@nb.njit(fastmath=True)
def σ_inv_e(e, A13):
    A23 = A13 ** 2
    return (e * (.76 + 1.93 / A13) + (1.66 / A23 - 0.05)) * pi * 2.89 * A23


@nb.njit(fastmath=True)
def R_nuc(A):
    return 1.2 * A ** (1/3)


@nb.njit(float64[:](float64[:], int64))
def round_njt(x, decimals):
    out = np.empty(x.shape[0])
    return np.round_(x, decimals, out)


def m_ex(A_f, Z_f, mass_data):
    A = np.array((A_f, A_f - 1))
    res = np.zeros_like(A)
    cf = np.array([1, -1])
    for i, el in enumerate(A):
        res[i] = mass_excess(A_f, Z_f, mass_data)
        res[i] = LSD(A_f, Z_f, 0) if isnan(res[i]) else res[i]
    res = res * cf
    return sum(res)


def mass_excess(A, Z, data):
    search = (data['A'] == int(A)) & (data['Z'] == int(Z))
    if not any(search):
        return LSD(A, Z, 0)
    return data[search].iat[0, 3]


@nb.njit(fastmath=True, nogil=True)
def fast_neck_finder(ρ_2, a, z, z_s, z_0):
    dρ2dz = aux.dρ2_jit(a, z, z_s, z_0, 1)
    dz = 4 * (z[1] - z[0])
    extr_list = []
    for i, el in enumerate(dρ2dz[:-1]):
        if copysign(1, el) != copysign(1, dρ2dz[i+1]):
            extr_list.append(i+1)
            if len(extr_list) > 1:
                dz_prime = z[extr_list[1] + 1] - z[extr_list[1] - 1]
                rn = sqrt(ρ_2[extr_list[1]]) if dz_prime <= dz else 0
                return extr_list[1], rn
    return extr_list[0], sqrt(ρ_2[extr_list[0]])


@nb.njit(fastmath=True, nogil=True)
def Wskpf_em_α_noZ(q, A, Z, E):

    N = 500
    
    R0 = 1.2 * A ** (1 / 3)

    a = aux.q_to_a(q)
    z_s, c = aux.c_z(a)

    z_full = np.linspace(z_s - c, z_s + c, 2 * N)
    z_full, ρ2_full = aux.ρ2_jit(a, z_full, z_s, c, True)

    z_nck_ind, _ = fast_neck_finder(ρ2_full, a, z_full, z_s, c)

    z_ar = (z_full, z_full[:z_nck_ind+1], z_full[z_nck_ind+1:])
    ρ_ar = (ρ2_full, ρ2_full[:z_nck_ind+1], ρ2_full[z_nck_ind+1:])

    # z_ar = (np.linspace(z_s - c, z_full[z_nck_ind], N),
    #         np.linspace(z_full[z_nck_ind + 1], z_s + c, N)) 

    α2 = np.zeros(3)
    z_c_ar = np.zeros(3)

    for i in range(3):
        z, ρ_2 = z_ar[i], ρ_ar[i]
        # aux.ρ2(a, z_ar[ii - 1], z_s, c, True) if ii != 0\
        #     else (z_full, ρ2_full)
        ρ = np.sqrt(ρ_2)
        z_c = aux.Sdx(ρ_2 * z, z) / aux.Sdx(ρ_2, z)
        z_c_ar[i] = z_c
        dif_z = z - z_c
        R = np.sqrt(ρ ** 2 + dif_z ** 2)

        vol = .75 * aux.Sdx(ρ_2, z)
        R_0_f = vol ** (1 / 3)

        integrand = (R/R_0_f - 1) ** 2 * (ρ_2 - dif_z * aux.dρ2_jit(a, z, z_s,
                                                                    c, 1)
                                          ) / R ** 3

        α2[i] = 2 * pi * aux.Sdx(integrand, z)
      
    vrat = np.array([1 - vol, vol])
    α2 /= R0

    bs, bk, bc = B_surf(α2[0]), B_curv(α2[0]), B_coul(α2[0])
    CN_b_cf = np.array((bc, bs, bk))

    bs1, bk1, bc1 = B_surf(α2[1]), B_curv(α2[1]), B_coul(α2[1])
    LF_b_cf = np.array((bc1, bs1, bk1))

    bs2, bk2, bc2 = B_surf(α2[2]), B_curv(α2[2]), B_coul(α2[2])
    RF_b_cf = np.array((bc2, bs2, bk2))

    cf = np.array([[bc1, bc2], [bs1, bs2], [bk1, bk2]])

    A_f = np.round(vrat[0] * A); A_f = np.array([A_f, A - A_f], dtype='i4')
    Z_f = np.round(vrat[0] * Z); Z_f = np.array((Z_f, Z - Z_f), dtype='i4')
   

    E_star_ff = density_deformed(A_f[1], Z_f[1], bs2, bk2, bc2)
    E_star_ff /= E_star_ff + density_deformed(A_f[0], Z_f[0], bs1, bk1, bc1)

    E_star_ff = np.array([1 - E_star_ff, E_star_ff])
    E_star_ff *= E
    
    LF_Z, RF_Z = Z_f.copy()
    LF_A_prime, RF_A_prime = A_f.copy()

    E_st = [List.empty_list(float64), List.empty_list(float64)]
    e_n = [List.empty_list(float64), List.empty_list(float64)]
    
    
    return e_n[0][0]

    for i in range(2):
        ΔB = cf[:, i] - np.ones(3)
        ΔE_def = ΔLSD_def(A_f[i], Z_f[i], ΔB)
        E_st[i].append(ΔE_def + m_exc_n(A_f[i], Z_f[i], mass_data_np)
                        + E_star_ff[i]) # T ** 2 * density(A_f[i], Z_f[i], *cf[:, i]))
        while E_st[i] > ΔM_n:
            ϵ_n_max = E_st[i][-1] - ΔM_n
            ϵ_n = np.linspace(0, ϵ_n_max, N)
            dϵ = ϵ_n[1]
            ϵ_n[1:] -= dϵ / 2
            coef = dϵ * fact / den_lvl_float(E_st[i][-1], A_f[i], Z_f[i])

            A_f[i] -= 1
            A13 = A_f[i] ** (1 / 3)
            f = σ_inv_e(ϵ_n[1:], A13) * den_lvl_arr(ϵ_n_max - ϵ_n[1:],
                                                    A_f[i], Z_f[i])
            f *= coef
            g = f.cumsum()
            grn = random.uniform(0, 1) * g[-1]
            if grn < g[1]:
                ϵ_neut = grn / g[0] * dϵ
            else:
                j = 2
                while g[j] < grn and j < N - 1:
                    j += 1
                ϵ_neut = ϵ_n[j] - dϵ * (g[j] - grn) / (g[j] - g[j-1])
            e_n[i].append(ϵ_neut)
            mass_ex = m_exc_n(A_f[i], Z_f[i], mass_data_np)
            E_st[i].append(ϵ_n_max - e_n[i] + mass_ex)

    return LF_Z, LF_A_prime, A_f[0], LF_b_cf,\
             RF_Z, RF_A_prime, A_f[1], RF_b_cf,\
             CN_b_cf, α2


    out = [[LF_Z, LF_A_prime, A_f[0], LF_b_cf, E_st[0], e_n[0]],
            [RF_Z, RF_A_prime, A_f[1], RF_b_cf, E_st[1], e_n[1]],
            CN_b_cf, α2]



    return [LF_Z, LF_A_prime, A_f[0], LF_b_cf, E_st[0], e_n[0]],\
           [RF_Z, RF_A_prime, A_f[1], RF_b_cf, E_st[1], e_n[1]],\
           CN_b_cf, α2


@nb.njit(fastmath=True, nogil=True)
def neck_break_energy(A_ar, Z_ar, B_surf_ar):
    A23_ar = A_ar ** (2/3)
    I_ar_term = 1 - 2.2938 * (1  - 2 * Z_ar / A_ar) ** 2
    return sum(16.9707 * I_ar_term * A23_ar * B_surf_ar * np.array([-1, 1, 1]))


@nb.njit(fastmath=True, nogil=True)
def ΔLSD_def(A, Z, ΔB):
    A13 = A ** (1/3)
    N = A - Z
    I = (N - Z) / A
    z_terms = 0.70978 / A13 * ΔB[0] * Z ** 2
    curv_terms = (16.9707 * (1 - 2.2938 * I ** 2) * A13 * ΔB[1]
                  + 3.8602 * (1 + 2.3764 * I ** 2) * ΔB[2]) * A13
    return z_terms + curv_terms


@nb.njit(fastmath=True, nogil=True)
def ΔE_def(A, Z, B_coul, B_surf, B_curv, m_data):
    return LSD_def(A, Z, B_coul, B_surf, B_curv) - m_ex_np_ver(A, Z, m_data)




def Wskpf_em_α_POM(q, A, Z, T, qlim, dq, N_q, R12, α2, vol):

    N = 500
    out = [{'Z': '', 'A_prime': '', 'α2': '', 'A_rest': '',
            'E*': '', 'e_n':[]} for i in range(2)]

    E_star = lvl_dens_param(A, Z) * T ** 2

    α_2 = np.array([gh_ap3d(q, qlim, dq, N_q, el) for el in α2])
    α_2[α_2 < 0] = 0
    
    R_12 = 1.2 * A ** (1 / 3) * gh_ap3d(q, qlim, dq, N_q, R12)
    
    vrat = np.array([gh_ap3d(q, qlim, dq, N_q, el) for el in vol])
    A_f = round(vrat[0] * A); A_f = np.array((A_f, A - A_f), dtype=int)
    A_h = max(A_f)

    hi = np.where(A_f == A_h)[0][0]
    li = np.array((0, 1))[np.array((0, 1)) != hi][0]

    Z_f = int(vrat[hi] * Z)
    l = 4
    Z_range = np.arange(Z_f - l, Z_f + l)

    E_diff = np.array([LSD(A_f[hi], z_f, α_2[hi]) + LSD(A_f[li], Z - z_f,
                                                        α_2[li])
                       + 1.44 * z_f * (Z - z_f) / R_12 - LSD(A, Z, )
                       for z_f in Z_range])

    min_E_diff = min(E_diff)

    E0 = 5

    Z_distr = np.exp(-((E_diff - min_E_diff) / E0) ** 2)
    int_Z_distr = Z_distr.cumsum()
    int_Z_distr /= int_Z_distr[-1]
    rdnum = random.uniform(0, 1)

    Z_f = int(Z_range[int_Z_distr > rdnum][0])

    Z_new = np.zeros(2, dtype=int)
    Z_new[hi] = Z_f; Z_new[li] = Z - Z_f
    Z_f = Z_new.copy()

    ϵ_n_max = []
    for i in range(2):
        ΔE_def = LSD(A_f[i], Z_f[i], α_2[i]) - LSD(A_f[i], Z_f[i], 0)
        # mass_ex = m_ex(A_f[i], Z_f[i], mass_data)
        mass_ex = mass_excess(A_f[i], Z_f[i], mass_data) -\
            mass_excess(A_f[i] - 1, Z_f[i], mass_data)
        out[i]['E*'] = ΔE_def + mass_ex + vrat[i] * E_star
        ϵ_n_max.append(out[i]['E*'])
    ϵ_n_max = np.array(ϵ_n_max)

    out[0]['α2'], out[1]['α2'] = α_2.copy()
    out[0]['Z'], out[1]['Z'] = Z_f.copy()
    out[0]['A_prime'], out[1]['A_rest'] = A_f.copy()
    out[0]['A_rest'], out[1]['A_prime'] = A_f.copy()

    out[0]['E*'], out[1]['E*'] = [[el, ] for el in ϵ_n_max]

    fact = 2 * m_unit / (pi2 * h_bar_c ** 2)

    for i in range(2):
        while ϵ_n_max[i] >= ΔM_n:
            ϵ_n_max[i] -= ΔM_n
            ϵ_n = np.linspace(0, ϵ_n_max[i], N)
            dϵ = ϵ_n[1] - ϵ_n[0]
            ϵ_n += dϵ / 2
            ϵ_n = ϵ_n[:-1]
            cf = dϵ * fact / den_level(out[i]['E*'][-1], out[i]['A_rest'],
                                       out[i]['Z'])

            out[i]['A_rest'] -= 1
            A13 = out[i]['A_rest'] ** (1 / 3)

            f = σ_inv_e(ϵ_n, A13) * den_level(ϵ_n_max[i] - ϵ_n, out[i]['A_rest'],
                                            out[i]['Z'])
            g = cf * f.cumsum()

            grn = random.uniform(0, 1) * g[-1]

            if abs(grn) < abs(g[1]):
                ϵ_neut = grn / g[1] * dϵ

            for j, ϵ in enumerate(ϵ_n):
                if g[j + 1] > grn:
                    ϵ_neut = ϵ - dϵ * (g[j + 1] - grn) / (g[j + 1] - g[j])
                    break

            out[i]['e_n'].append(ϵ_neut)
            # mass_ex = m_ex(out[i]['A_rest'], out[i]['Z'], mass_data)
            mass_ex = mass_excess(out[i]['A_rest'], out[i]['Z'], mass_data) -\
                mass_excess(out[i]['A_rest'] - 1, out[i]['Z'], mass_data)
            ϵ_n_max[i] -= out[i]['e_n'][-1]
            out[i]['E*'].append(ϵ_n_max[i])

    return out


# @nb.njit(fastmath=True, nogil=True)
def Wskpf_em_α(q, A, Z, T, qlim, dq, N_q, R12, α2, vol):

    N = 1000
    limit = N -1

    out = [{'Z': '', 'A_prime': '', 'α2': '', 'A_rest': '', 'E*': '', 'e_n':[]}
            for i in range(2)]

    E_star = lvl_dens_param(A, Z) * T ** 2

    α_2 = np.array([gh_ap3d(q, qlim, dq, N_q, el) for el in α2])
    α_2[α_2 < 0] = 0
    R_12 = 1.2 * A ** (1 / 3) * gh_ap3d(q, qlim, dq, N_q, R12)
    vrat = np.array([gh_ap3d(q, qlim, dq, N_q, el) for el in vol])
    A_f = round(vrat[0] * A); A_f = np.array((A_f, A - A_f), dtype=int)
    A_h = max(A_f)
    hi = np.where(A_f == A_h)[0][0]
    li = np.array((0, 1))[np.array((0, 1)) != hi][0]
    Z_f = int(vrat[hi] * Z)
    l = 4
    Z_range = np.arange(Z_f - l, Z_f + l)

    # EZ = np.array([mass_excess(A_h, i, mass_data) for i in Z_range])
    # while len(EZ[np.isnan(EZ)]) > l - 1:
    #     Z_f += l if len(EZ[l:][np.isnan(EZ[l:])]) < len(EZ[:l][np.isnan(EZ[:l])])\
    #         else -l
    #     Z_range = np.arange(Z_f - l, Z_f + l)
    #     EZ = np.array([mass_excess(A_h, i, mass_data) for i in Z_range])
    # Z_range = Z_range[EZ != np.nan]

    E_diff = np.array([LSD(A_f[hi], z_f, α_2[hi]) + LSD(A_f[li], Z - z_f,
                                                        α_2[li])
                       + 1.44 * z_f * (Z - z_f) / R_12 - mass_excess(A, Z,
                                                                     mass_data)
                       for z_f in Z_range])

    min_E_diff = min(E_diff)

    E0 = 5

    Z_distr = np.exp(-((E_diff - min_E_diff) / E0) ** 2)
    int_Z_distr = Z_distr.cumsum()
    int_Z_distr /= int_Z_distr[-1]
    random.seed()
    rdnum = random.uniform(0, 1)

    Z_f = int(Z_range[int_Z_distr > rdnum][0])

    Z_new = np.zeros(2, dtype=int)
    Z_new[hi] = Z_f; Z_new[li] = Z - Z_f
    Z_f = Z_new.copy()

    ϵ_n_max = []
    for i in range(2):
        ΔE_def = LSD(A_f[i], Z_f[i], α_2[i]) - LSD(A_f[i], Z_f[i], 0)
        # mass_ex = m_ex(A_f[i], Z_f[i], mass_data)
        mass_ex = mass_excess(A_f[i], Z_f[i], mass_data) -\
            mass_excess(A_f[i] - 1, Z_f[i], mass_data)
        out[i]['E*'] = ΔE_def + mass_ex + vrat[i] * E_star
        ϵ_n_max.append(out[i]['E*'])
    ϵ_n_max = np.array(ϵ_n_max)

    out[0]['α2'], out[1]['α2'] = α_2.copy()
    out[0]['Z'], out[1]['Z'] = Z_f.copy()
    out[0]['A_prime'], out[1]['A_rest'] = A_f.copy()
    out[0]['A_rest'], out[1]['A_prime'] = A_f.copy()

    out[0]['E*'], out[1]['E*'] = [[el, ] for el in ϵ_n_max]

    fact = 2 * m_unit / (pi2 * h_bar_c ** 2)

    for i in range(2):
        while ϵ_n_max[i] > ΔM_n:
            ϵ_n_max[i] -= ΔM_n
            ϵ_n = np.linspace(0, ϵ_n_max[i], N)
            dϵ = ϵ_n[1]
            ϵ_n[1:] -= dϵ / 2
            coef = dϵ * fact / den_level(out[i]['E*'][-1], out[i]['A_rest'],
                                         out[i]['Z'])

            out[i]['A_rest'] -= 1
            A13 = out[i]['A_rest'] ** (1 / 3)

            f = σ_inv_e(ϵ_n, A13) * den_level(ϵ_n_max[i] - ϵ_n, out[i]['A_rest'],
                                              out[i]['Z'])
            f *= coef
            g = f.cumsum()

            grn = random.uniform(0, 1) * g[-1]

            if abs(grn) < abs(g[1]):
                ϵ_neut = grn / g[1] * dϵ

            if grn < g[1]:
                ϵ_neut = grn / g[0] * dϵ
            else:
                j = 2
                while g[j] < grn and j < N - 1:
                    j += 1
                ϵ_neut = ϵ_n[j] - dϵ * (g[j] - grn) / (g[j] - g[j-1])

            out[i]['e_n'].append(ϵ_neut)
            # mass_ex = m_ex(out[i]['A_rest'], out[i]['Z'], mass_data)
            mass_ex = mass_excess(out[i]['A_rest'], out[i]['Z'], mass_data) -\
                mass_excess(out[i]['A_rest'] - 1, out[i]['Z'], mass_data)
            ϵ_n_max[i] -= out[i]['e_n'][-1]
            out[i]['E*'].append(ϵ_n_max[i])

    return out


def Wskpf_em_def(q, A, Z, T, qlim, dq, N_q, R12, vol):

    N = 1000

    out = [{'Z': '', 'A_prime': '', 'A_rest': '', 'B_cf': '', 'E*': [],
            'e_n':[]} for i in range(2)] + [[],]

    bc, bs, bk, bc1, bs1, bk1, bc2, bs2, bk2 = aux.surface_coefficients(q)
    out[0]['B_cf'] = bc1, bs1, bk1
    out[1]['B_cf'] = bc2, bs2, bk2
    out[2] = bc, bs, bk

    cf = np.array([[bc1, bc2], [bs1, bs2], [bk1, bk2]])

    E_star = density(A, Z, *out[2]) * T ** 2

    R_12 = 1.2 * A ** (1 / 3) * gh_ap3d(q, qlim, dq, N_q, R12)
    vrat = np.array([gh_ap3d(q, qlim, dq, N_q, el) for el in vol])
    A_f = round(vrat[0] * A); A_f = np.array((A_f, A - A_f), dtype=int)
    # Z_f = round(vrat[0] * Z); Z_f = np.array((Z_f, Z - Z_f), dtype=int)

    A_h = max(A_f)
    hi = np.where(A_f == A_h)[0][0]
    li = np.array((0, 1))[np.array((0, 1)) != hi][0]
    Z_f = int(vrat[hi] * Z)

    lim = 4
    Z_range = np.arange(Z_f - lim, Z_f + lim)

    cf_h = tuple(cf[:, hi])
    cf_l = tuple(cf[:, li])

    E_diff = np.array([LSD_def(A_f[hi], z_f, *cf_h)
                        + LSD_def(A_f[li], Z - z_f, *cf_l)
                        + 1.44 * z_f * (Z - z_f) / R_12 for z_f in Z_range]
                      )
    E_diff -= LSD_def(A, Z, *out[2])

    min_E_diff = min(E_diff)                            

    E0 = 5

    Z_distr = np.exp(-((E_diff - min_E_diff) / E0) ** 2)
    int_Z_distr = Z_distr.cumsum()
    rdnum = random.uniform(0, 1) * int_Z_distr[-1]

    Z_f = int(Z_range[int_Z_distr > rdnum][0])

    Z_new = np.zeros(2, dtype=int)
    Z_new[hi], Z_new[li] = Z_f, Z - Z_f
    Z_f = Z_new.copy()

    for i in range(2):
        ΔE_def = LSD_def(A_f[i], Z_f[i], *(cf[:, i] - np.ones(3)))
        # mass_ex = m_ex(A_f[i], Z_f[i], mass_data)
        mass_ex = m_ex_np_ver(A_f[i], Z_f[i], mass_data_np) -\
            m_ex_np_ver(A_f[i] - 1, Z_f[i], mass_data_np)
        out[i]['E*'].append(ΔE_def + mass_ex + vrat[i] * E_star)

    out[0]['Z'], out[1]['Z'] = Z_f.copy()
    out[0]['A_prime'], out[1]['A_rest'] = A_f.copy()
    out[0]['A_rest'], out[1]['A_prime'] = A_f.copy()

    for i in range(2):
        while out[i]['E*'][-1] > ΔM_n:
            out[i]['E*'].append(out[i]['E*'][-1] - ΔM_n)
            ϵ_n_max = out[i]['E*'][-1]
            ϵ_n = np.linspace(0, ϵ_n_max, N)
            dϵ = ϵ_n[1]
            ϵ_n[1:] -= dϵ / 2
            coef = dϵ * fact / den_level(out[i]['E*'][-1], out[i]['A_rest'],
                                         out[i]['Z'])

            out[i]['A_rest'] -= 1
            A13 = out[i]['A_rest'] ** (1 / 3)
            f = σ_inv_e(ϵ_n[1:], A13) * den_level(ϵ_n_max - ϵ_n[1:],
                                                  out[i]['A_rest'],
                                                  out[i]['Z'])
            f *= coef
            g = f.cumsum()
            grn = random.uniform(0, 1) * g[-1]
            if grn < g[1]:
                ϵ_neut = grn / g[0] * dϵ
            else:
                j = 2
                while g[j] < grn and j < N - 1:
                    j += 1
                ϵ_neut = ϵ_n[j] - dϵ * (g[j] - grn) / (g[j] - g[j-1])
            out[i]['e_n'].append(ϵ_neut)
            mass_ex = m_ex_np_ver(out[i]['A_rest'], out[i]['Z'], mass_data_np)\
                - m_ex_np_ver(out[i]['A_rest'] - 1, out[i]['Z'], mass_data_np)
            out[i]['E*'][-1] += mass_ex - out[i]['e_n'][-1]

    # import seaborn as sns
    # sns.histplot(g/g[-1], binwidth=.05, stat='probability')

    return out


def debugger_out_dat(out_lib, vrat, A, Z, T, cf):
    out_lib[0]['Z'], out_lib[1]['Z'] = np.round(vrat * Z).astype(int)
    A_f = np.round(vrat[0] * A)
    A_f = np.array([A_f, A - A_f], dtype=int)
    out_lib[0]['A_prime'], out_lib[1]['A_rest'] = A_f.copy()
    out_lib[0]['A_rest'], out_lib[1]['A_prime'] = A_f.copy()
    h_idx = np.where(A_f == max(A_f))[0][0]
    out_lib[h_idx]['E*'].append(T ** 2 * density(A, Z, *cf[:, h_idx]))
    return out_lib


def Wskpf_em_def_new(q, A, Z, T):

    N = 200

    out = [{'Z': '', 'A_prime': '', 'A_rest': '', 'B_cf': '', 'E*': [],
            'ϵ_max': [], 'e_n':[]} for i in range(2)] + [[],[]]

    a = q_into_a(q)
    c, z_s = c_z(a)

    R_0 = 1
    z_0 = c * R_0

    z_full = np.linspace(z_s - z_0, z_s + z_0, 2 * N)
    z_full, ρ2_full = aux.ρ2_jit(a, z_full, z_s, z_0, True)

    z_nck_ind, r_n = fast_neck_finder(ρ2_full, a, z_full, z_s, z_0)

    # coul_en = 1
    # coul_en = .5 if r_n <= 0 else 1

    z_ar = [np.linspace(z_s - z_0, z_full[z_nck_ind - 1], N),
            np.linspace(z_full[z_nck_ind], z_s + z_0, N)]

    ρ_2_f = []
    z_f = []

    for el in z_ar:
       z, ρ_2 = aux.ρ2_jit(a, el, z_s, z_0, True)
       ρ_2_f.append(ρ_2)
       z_f.append(z)

    v_r = np.zeros(2)
    z_c = np.zeros(2)

    for i, ρ_2, z in zip(range(2), ρ_2_f, z_f):
        v_r[i] = .75 * aux.Sdx(ρ_2, z)
        if v_r[i] > 1 or v_r[i] <= 0:
            return debugger_out_dat(out, np.array([v_r[i], 1 - v_r[i]]), 
                                    A, Z, T, np.ones((3, 2)))
        z_c[i] = aux.Sdx(ρ_2 * z, z) / aux.Sdx(ρ_2, z)
    r12 = z_c[1] - z_c[0]

    if abs(v_r.sum() - 1) > N ** (-4/3):
        R_0 = v_r.sum() ** (-1 / 3)
        v_r /= v_r.sum()
        z_ar = [el * R_0 for el in z_ar]

    R_sph_f = [R_0 * el ** (1 / 3) for el in v_r]

    a_l_new = aux.a_trfrm(ρ2_full, z_full, z_nck_ind,
                            R_sph_f[0], 'left', False)
    a_r_new = aux.a_trfrm(ρ2_full, z_full, z_nck_ind,
                            R_sph_f[1], 'right', False)
    q_l_new, q_r_new = aux.a_to_q(a_l_new), aux.a_to_q(a_r_new)

    bc1, bs1, bk1 = aux.fcs_short(q_l_new)
    bc2, bs2, bk2 = aux.fcs_short(q_r_new)
    bc, bs, bk = aux.fcs_short(q)

    # bc, bs, bk, bc1, bs1, bk1, bc2, bs2, bk2 = aux.surface_coefficients(q)

    out[0]['B_cf'] = bc1, bs1, bk1
    out[1]['B_cf'] = bc2, bs2, bk2
    out[2] = bc, bs, bk

    cf = np.array([[bc1, bc2],
                   [bs1, bs2],
                   [bk1, bk2]])

    R_12 = 1.2 * A ** (1 / 3) * r12
    vrat = v_r.copy()
    A_f = np.round(vrat[0] * A)
    if A_f - A >= 0:
        return debugger_out_dat(out, vrat, A, Z, T, cf)
    
    A_f = np.array([A_f, A - A_f], dtype=int)
    
    # Z_f = round(vrat[0] * Z); Z_f = np.array((Z_f, Z - Z_f), dtype=int)

    A_h = max(A_f)
    hi = np.where(A_f == A_h)[0][0]
    li = np.array((0, 1))[np.array((0, 1)) != hi][0]
    
    Z_f = np.int32(vrat[hi] * Z)

    lim = 6
    Z_range = np.arange(Z_f - lim, Z_f + lim)
    Z_range = Z_range[Z_range < Z]

    cf_h = tuple(cf[:, hi])
    cf_l = tuple(cf[:, li])

    # EZ = np.array([mass_excess(A_h, i, mass_data) for i in Z_range])
    # while len(EZ[np.isnan(EZ)]) > l - 1:
    #     Z_f += l if len(EZ[l:][np.isnan(EZ[l:])]) < len(EZ[:l][np.isnan(EZ[:l])])\
    #         else -l
    #     Z_range = np.arange(Z_f - l, Z_f + l)
    #     EZ = np.array([mass_excess(A_h, i, mass_data) for i in Z_range])
    # Z_range = Z_range[EZ != np.nan]

    E_Coul = .72 * (Z**2 / A**(1/3) * bc - Z_range**2 / A_h**(1/3) * cf[0, hi]
                    - (Z_range - Z)**2 / (A - A_h)**(1/3) * cf[0, li])

    E_diff = np.array([LSD_def(A_f[hi], z_f, *cf_h)
                       + LSD_def(A_f[li], Z - z_f, *cf_l) for z_f in Z_range]
                      ) + 1.44 * (Z - Z_range) * Z_range / R_12
                    
    # E_diff = np.array([LSD_def(A_f[hi], z_f, *cf_h)
    #                    + LSD_def(A_f[li], Z - z_f, *cf_l) for z_f in Z_range]
    #                   ) + E_Coul


    # E_diff -= LSD_def(A, Z, *out[2])
    E_diff -= LSD(A, Z)

    min_E_diff = min(E_diff)

    E0 = 5

    Z_distr = np.exp(-((E_diff - min_E_diff) / E0) ** 2)

    int_Z_distr = Z_distr.cumsum()
    rdnum = random.uniform(0, 1) * int_Z_distr[-1]

    Z_f = int(Z_range[int_Z_distr > rdnum][0])

    # out[-1].append(coul_en * E_Coul[Z_range == Z_f][-1])
    out[-1].append(E_Coul[Z_range == Z_f][0])
    out[-1].append(R_12)
    out[-1].append(r_n)
    out[-1].append(8.48535 * (A ** (1/3) * r_n) ** 2)
    
    Z_new = np.zeros(2, dtype=np.int64)
    Z_new[hi], Z_new[li] = Z_f, Z - Z_f
    Z_f = Z_new.copy()
    
    # E_neck = neck_break_energy(np.array([A, A_f[0], A_f[1]]),
    #                            np.array([Z, Z_f[0], Z_f[1]]),
    #                            np.array([bs, bs1, bs2])
    #                            )
    # if E_neck < 0:
    #     out[-1][0] += coul_en * E_neck

    out[0]['Z'], out[1]['Z'] = Z_f.copy()
    out[0]['A_prime'], out[1]['A_rest'] = A_f.copy()
    out[0]['A_rest'], out[1]['A_prime'] = A_f.copy()
    
    a_rat = lvl_dens_param(A_f[li], Z_f[li]) / lvl_dens_param(A_f[hi], Z_f[hi])

    E_star_CN = T ** 2 * density(A, Z, bc, bs, bk)

    # E_add = np.zeros(2)
    # E_add[li] = out[-1][-1] / (1 + a_rat)
    # E_add[hi] = out[-1][-1] - E_add[li]
    # E_star_i = E_star_CN * np.array([e_l_cf, e_r_cf])
    # E_star_i = (E_add + E_star_CN) * np.array([e_l_cf, e_r_cf])

    E_star_i = np.zeros(2)
    E_star_i[li] = E_star_CN / (1 + a_rat)
    E_star_i[hi] = E_star_CN - E_star_i[li]
    
    for i in range(2):
        
        # ΔE = ΔLSD_def(A_f[i], Z_f[i], *(cf[:, i] - np.ones(3)))
        E_star_tot_f = E_star_i[i] + ΔE_def(A_f[i], Z_f[i], *cf[:, i],
                                            mass_data_np)
        # ΔE = m_exc_n(A_f[i], Z_f[i], mass_data_np)
        ΔE_exc = m_exc_n(out[i]['A_prime'], out[i]['Z'], mass_data_np) - ΔM_n
        out[i]['E*'].append(E_star_tot_f)
        ϵ_n_max = ΔE_exc + E_star_tot_f
        out[i]['ϵ_max'].append(ϵ_n_max)
        if E_star_tot_f < n_emission:
            continue

        # mass_ex = m_ex(A_f[i], Z_f[i], mass_data)
        # out[i]['E*'].append(ΔE + E_0 + T ** 2 * density(A_f[i],
        #                                                  Z_f[i], *cf[:, i]))


        # out[i]['E*'].append(ΔE + m_exc_n(A_f[i], Z_f[i], mass_data_np)
        #                     + T ** 2 * density(A_f[i], Z_f[i], *cf[:, i]))

        # out[i]['E*'].append(ΔE + m_exc_n(A_f[i], Z_f[i], mass_data_np) + E_add[i]
        #                     + T ** 2 * density(A_f[i], Z_f[i], *cf[:, i]))

    

    # for i in range(2):
        # while out[i]['E*'][-1] >= ΔM_n:
        #     ϵ_n_max = out[i]['E*'][-1] - ΔM_n

        while ϵ_n_max > 0:
            ϵ_n = np.linspace(0, ϵ_n_max, N)
            dϵ = ϵ_n[1]
            ϵ_n[1:] -= dϵ / 2
            coef = dϵ * fact / den_lvl_float(out[i]['ϵ_max'][-1], out[i]['A_rest'],
                                             out[i]['Z'])

            out[i]['A_rest'] -= 1
            A13 = out[i]['A_rest'] ** (1 / 3)
            f = σ_inv_e(ϵ_n[1:], A13) * den_lvl_arr(ϵ_n_max - ϵ_n[1:],
                                                    out[i]['A_rest'],
                                                    out[i]['Z'])
            f *= coef
            g = f.cumsum()
            grn = random.uniform(0, 1) * g[-1]
            if grn < g[1]:
                ϵ_neut = grn / g[0] * dϵ
            else:
                j = 2
                while g[j] < grn and j < N - 1:
                    j += 1
                ϵ_neut = ϵ_n[j] - dϵ * (g[j] - grn) / (g[j] - g[j-1])
            out[i]['e_n'].append(ϵ_neut)
            mass_ex = m_exc_n(out[i]['A_rest'], out[i]['Z'], mass_data_np) - ΔM_n
            ϵ_n_max -= ϵ_neut - mass_ex
            out[i]['ϵ_max'].append(ϵ_n_max)
    # import seaborn as sns
    # sns.histplot(g/g[-1], binwidth=.05, stat='probability')

    return out


@nb.njit(fastmath=True, nogil=True)
def m_exc_n(A, Z, ms_data):
    return m_ex_np_ver(A, Z, ms_data) - m_ex_np_ver(A - 1, Z, ms_data)


@nb.njit(fastmath=True, nogil=True)
def m_exc_v(A, Z, ms_data, v):
    if v in ['p', 'H', 'proton']:
        A_d = A - 1
        Z_d = Z - 1
        Δ = ΔM_p
    elif v in ['alpha', 'He', 'α']:
        A_d = A - 4
        Z_d = Z - 2
        Δ = ΔM_α
    else:
        A_d = A - 1
        Z_d = Z
        Δ = ΔM_n
    return m_ex_np_ver(A, Z, ms_data) - m_ex_np_ver(A_d, Z_d, ms_data) - Δ


@nb.njit(fastmath=True, nogil=True)
def Wskpf_em_CN(q, A, Z, E_star, def_cf, gs_def):

    N = 200
    mass_ex = m_exc_n(A, Z, mass_data_np)
    Δb_cl, Δb_sf, Δb_cr  = def_cf - gs_def
    # E_exc_tot = ΔLSD_def(A, Z, Δb_cl, Δb_sf, Δb_cr) + mass_ex + E_star
    E_exc_tot = ΔE_def(A, Z, *def_cf, mass_data_np) + mass_ex + E_star
    # neut_ϵ = []
    ϵ_neut = 0

    if E_exc_tot > ΔM_n:
        ϵ_max = E_exc_tot - ΔM_n
        ϵ_n = np.linspace(0, ϵ_max, N)
        dϵ = ϵ_n[1]
        ϵ_n[1:] -= dϵ / 2
        coef = dϵ * fact / den_lvl_float(E_exc_tot, A, Z)
        A -= 1
        A13 = A ** (1 / 3)
        f = σ_inv_e(ϵ_n[1:], A13) * den_lvl_arr(E_exc_tot - ϵ_n[1:], A, Z)
        f *= coef
        g = f.cumsum()
        grn = random.uniform(0, 1) * g[-1]
        if grn < g[1]:
            ϵ_neut = grn / g[0] * dϵ
        else:
            j = 2
            while g[j] < grn and j < N - 1:
                j += 1
            ϵ_neut = ϵ_n[j] - dϵ * (g[j] - grn) / (g[j] - g[j-1])
        E_exc_tot = ϵ_max - ϵ_neut

    return A, E_exc_tot, ϵ_neut


def neck_finder(ρ_2, z):
    dρ2dz = np.gradient(ρ_2, z)
    extr_list = []
    dz =  4 * (z[1] - z[0])
    for i, el in enumerate(dρ2dz[:-1]):
        if copysign(1, el) != copysign(1, dρ2dz[i+1]):
            extr_list.append(i)
            if len(extr_list) > 1:
                dz_prime = z[extr_list[1]+1] - z[extr_list[1] -1]
                rn = sqrt(ρ_2[extr_list[1]]) if dz_prime <= dz else 0
                return extr_list[1], rn
    return extr_list[0], sqrt(ρ_2[extr_list[0]])


def neck_finder2(ρ_2, z):
    N = len(z) // 2
    dz = np.round(np.diff(z), 5)
    dz_min, dz_max = 2 * min(dz), max(dz)
    if dz_min < dz_max:
        ind = np.where(max(np.diff(z)) == np.diff(z))[0]
        if ind.size != 0 and N // 4 <= ind[0] <= 7 * N // 4:
            return ind[0]
    dρ2dz = np.gradient(ρ_2, z)
    abs_dρ2dz = np.abs(dρ2dz)
    ind_dρ2 = argrelextrema(abs_dρ2dz, np.less)[0]
    if len(z) % 2 != 1 and all(np.isclose(0, abs_dρ2dz[:N]
                                          - np.flip(abs_dρ2dz[N:]))):
        return N
    d2ρ2dz2 = np.gradient(dρ2dz, z)
    eps = - max(np.abs(np.diff(d2ρ2dz2)))

    ind_d2ρ2 = np.where(d2ρ2dz2 >= eps)[0]
    ind_d2ρ2 = ind_d2ρ2[(ind_d2ρ2 > N // 4) & (ind_d2ρ2 < 7 * N // 4)]
    ind = np.intersect1d(ind_dρ2, ind_d2ρ2)
    if ind.size == 0:
        ind = np.where(abs_dρ2dz < - eps)[0]
        if ind_d2ρ2.size != 0 and ind.size != 0:
            ind = np.intersect1d(ind_dρ2, ind)
        else:
            return N
    return ind[0]


###############################################################################

m_unit = 939.5656
h_bar_c = 197.327
sqrt_pi = 1.7724538509055159
pi2 = 9.869604401089358

fact = 2 * m_unit / (pi2 * h_bar_c ** 2)

mass_data = pd.read_fwf('mass 16.txt')[['A', 'Z', 'N', 'M_EXS']]
for i, el in enumerate(mass_data.loc[:, 'M_EXS']):
    if '#' in el:
        mass_data.loc[i, 'M_EXS'] = el.split('#')[0]
mass_data['M_EXS'] = mass_data['M_EXS'].astype(np.float32) / 1e3
mass_data_np = mass_data.to_numpy()
ΔM_n = mass_excess(1, 0, mass_data)
ΔM_p = mass_excess(1, 1, mass_data)
ΔM_α = mass_excess(4, 2, mass_data)

# colspecs = [(0, 3), (4, 7), (18, 30)]
# data_nubase = pd.read_fwf('nubase_mas20.txt', colspecs=colspecs,
#                           names=['A', 'Z', 'M_EXS']).dropna()
# data_nubase.insert(2, 'N', data_nubase['A'] - data_nubase['Z'])
# data_nubase['M_EXS'] = data_nubase['M_EXS'].str.replace('#','').astype(np.float32) / 1e3
# mass_data_NUB_np = data_nubase.to_numpy()

n_emission = 6

E_0 = 1
E_γ_thres = 5
ΔM_n_γ = ΔM_n + E_γ_thres

if __name__ == "__main__":
    sys.exit()
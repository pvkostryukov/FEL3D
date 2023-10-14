import os
import sys
import math
from math import pi, sqrt, copysign, sin, cos

import numpy as np
import numba as nb

###############################################################################
############################# CONSTANTS VALUES ################################
###############################################################################
a2_0 = 1.0320491
a4_0 = .03822404
a6_0 = .00825639
a8_0 = 0.00300889
###############################################################################
m_unit = 939.5656
h_bar_c = 197.327
R_cf = 1.2

mtens_coef = 75 / pi * m_unit / h_bar_c ** 2 * R_cf ** 2
###############################################################################
ng1 = 48
ng2 = 16
ng3 = 32

xg1, wg1 = gauleg(ng1)
xg2, wg2 = gauleg(ng2)
xg3, wg3 = gauleg(ng3)

Δφ = np.zeros((ng2, ng3))
φ1 = .25 * pi * (xg2 + 1)
φ2 = pi * (xg3 + 1)
for i, j in np.ndindex(Δφ.shape):
    Δφ[i, j] = φ1[i] - φ2[j]

cosΔφ = np.cos(Δφ)
sinΔφ = np.sin(Δφ)

###############################################################################
############################# SOME MATH FUNCTIONS #############################

@nb.njit(fastmath=True)
def cum_trapz(func, x):
    Δx = np.diff(x)
    cum_f = np.zeros_like(Δx)
    for i, el in enumerate(Δx):
        cum_f[i] = cum_f[i - 1] + .5 * el * (func[i+1] + func[i])
    return cum_f


@nb.njit(fastmath=True)
def Sdx(f, x):
    """One dimensional integration by Simpson-like formula
       f  array of function values f(x)
       x  array of variables
    """
    dx = abs(x[1] - x[0])
    f_odd = 4 * sum(f[1:-1][::2])
    f_even = 2 * sum(f[2:-1][::2])
    return dx / 3 * (f[0] + f[-1] + f_odd + f_even)


@nb.njit
def gauleg(n):

    x = np.zeros(n)
    w = np.zeros(n)
#
#   Nodes and weights for Gauss-Legendre integration
#
    m = n // 2
    z1 = 0

    z = np.cos(pi * (np.arange(1, m + 1) - .25) / (n + .5))

    for i, el in enumerate(z):
#
#   Rough estimate of the node position
#
        while abs(z[i] - z1) > 1e-16:
            pa = 1
            pb = 0
            for j in range(1, n + 1):
                pc = pb
                pb = pa
                pa = ((2 * j - 1) * el * pb - (j-1) * pc) / j
            pl = n * (el * pa - pb) / (el**2 - 1)
            z1 = el
            el -= pa / pl
        x[i] = -el
        x[-i - 1] = el
        w[i] = 2 / ((1 - el ** 2) * pl ** 2)
        w[-i - 1] = w[i]

    return x, w

###############################################################################

@nb.njit(fastmath=True)
def q_to_a(q):
    """ Definition of the Fourier expansion coefficients a_n.
        In addition to the a_n coefficients a_n, one introduces new collective
        coordinates q_n that are linear combinations of the a_n.
        This is done in such a way that the q_n all vanish for a spherical shape, 
        and that, except for the elongation parameter q_2, they remain small
        all along the liquid-drop path to fission(q_4 = q_6 = q_8 = 0 at the LD saddle).
        These new collective coordinates q_n are defined as:
           q_1 = eta = (b-a)/(b+a)       non-axiality parameter with a,b the half axis
           
           q_2 = a^o_2/a_2 - a_2/a^o_2   elongation q_2>0 prolate, q_2<0 oblate
           
           q_4 = a_4 + [(q_2/9)^2 + (a^o_4)^2]^(1/2)
           
           q_6 = a_6 - [(q_2/100)^2 + (a^o_6)^2]^(1/2)
           
           q_8 = a_8 + [(q_2/300)^2 + (a^o_8)^2]^(1/2)
        where a^o_n are the Fourier coefficients for a spherical shape.
        The parameters
           q_3 = a_3, q_5 = a_5 - (q_2 - 2) / 10 * a_3,
        are chosen such that the slope of the LD energy in the left-right
        asymmetry direction is the smallest possible.

        Add higher-order terms in the conditions below when ndim > 10.

        An analysis of the nuclear LDM PES suggests that the s[n] parameters
        used to define the q_{2n} deformation parameter are best chosen as
        s(4)=300., and s(5)=600.
    """
    dim = len(q)
    if dim >= 3:
        a = np.zeros(6)
        qs = np.zeros(6)
    
    if dim == 3:
        qs[1:dim + 1] = q.copy()
    else:
        qs[:dim] = q.copy()
    
    a[0] = qs[0]
    a[1] = .5 * a2_0 * (sqrt(qs[1] ** 2 + 4) - qs[1])
    a[2] = qs[2]
    a[3] = qs[3] - sqrt(a4_0 ** 2 + qs[1] ** 2 / 81)
    a[4] = qs[4] + .1 * (qs[1] - 2) * a[2]
    a[5] = qs[5] + sqrt(qs[1] ** 2 * 1e-4 + a6_0 ** 2)

    return a


@nb.njit(fastmath=True)
def q_into_alpha(q: np.array):
    """
    Convertion q coordinate into mass asymmetri coefficient via formula (A23) from
    C. Schmitt et al. (PRC 95, 034612 (2017)) paper
    """
    q_len = len(q.shape)
    a = np.zeros((len(q), 6)) if len(q.shape) > 1 else np.zeros(6)
    if q_len == 1:
        a = q_to_a(q)
        return a[3] / (a[2] - a[4] / 3) * (1 - 2*(a[2] + a[4]) / (a[2] + 9*a[4]))
    elif q_len > 1:
        a[:, 2] = .5 * a2_0 * (np.sqrt(q[:, 0] ** 2 + 4) - q[:, 0])
        a[:, 3] = q[:, 1]
        a[:, 4] = q[:, 2] - np.sqrt(a4_0 ** 2 + q[:, 0] ** 2 / 81)
        a[:, 5] = (q[:, 0] - 2) * a[:, 3] * .1
        return a[:, 3] / (a[:, 2] - a[:, 4] / 3) * (1 - 2*(a[:, 2] + a[:, 4])
                                                     / (a[:, 2] + 9*a[:, 4]))
    else:
        print('Error in dimentions of q vector coordinate')


@nb.njit(fastmath=True)
def a_to_q(a):
    dim = len(a)

    if dim >= 3:
        a_s = np.zeros(dim)
        qs = np.zeros(3)

        if dim == 3:
            a_s[1:dim + 1] = a.copy()
        else:
            a_s[:dim] = a.copy()

        qs[0] = a2_0 / a_s[1] - a_s[1] / a2_0 
        qs[1] = a_s[2]
        qs[2] = a_s[3] + sqrt(a4_0 ** 2 + qs[0] ** 2 / 81)

        return qs


@nb.njit(fastmath=True)
def a_trfrm(ρ_2, z, n_ind, R, lr, q_flag):
    """ For given prefragment shape transform to a or q (by flag q_flag) by
        Fourier tranform method, whch can be written
            a_n = R_f^(-2) S ρ2(u) cos|sin [(i - 0.5)πu] du
        where ρ2(u) is quadratic values of shape  
            u variates from - 1 to 1
            R_f radius of fission pre-fragment in spherical case
        lr chooses which pre-fragment you need (left/l or right/r).
    """
    a_dim = 6
    a_new = np.zeros(a_dim)
    n_pi = .5 * pi * np.arange(1, a_dim)
    if lr in ('left', 'l'):
        z_c = z[:n_ind]
        ρ_2_s = ρ_2[:n_ind]
    else:
        z_c = z[n_ind:]
        ρ_2_s = ρ_2[n_ind:]
    u = np.linspace(-1, 1, len(z_c))
    cos_sin = np.empty((len(n_pi), len(u)))
    for i, el in enumerate(n_pi):
        cos_sin[i] = np.cos(u * el) if i % 2 == 0 else np.sin(u * el)
    a_new[1:] = np.array([Sdx(ρ_2_s * el, u) for el in cos_sin]) / R ** 2
    return a_to_q(a_new) if q_flag else a_new


@nb.njit(fastmath=True)
def da_dq(q, a):
    dim = len(q)
    if dim >= 3:
        a2_0 = 1.03204910
        a4_0 = 0.03822404
        a6_0 = 0.00825639

        dadq = np.diag(np.ones_like(a))

        qs = np.zeros(7)
        if dim == 3:
            qs[1:dim + 1] = q.copy()
        else:
            qs[:dim] = q.copy()
        if len(a) >= 5:
            dadq[4, 1] = .1 * qs[2]
            dadq[4, 2] = .1 * (qs[1] - 2)

        s_cf = np.array([9, 100]) ** 2
        a0 = (a4_0, a6_0)

        dadq[1, 1] = .5 * (qs[1] / sqrt(qs[1]**2 + 4) - 1) * a2_0
        dadq[3::2, 1] = np.array([(-1) ** (i + 1) * qs[1] \
                                  / sqrt(qs[1]**2 / s_cf[i]
                                         + a0[i]**2) / s_cf[i]
                                  for i in range(2)
                                  ]
                                 )
        return dadq


@nb.njit(fastmath=True)
def p_m_ones(N):
    a = np.ones(N, dtype=nb.int8)
    a[::2] = -1
    return a


@nb.njit(fastmath=True)
def c_z(a):
    a_ev = a[1::2]
    a_odd = a[2::2]
    a_ev_l = len(a_ev)
    a_odd_l = len(a_odd)
    
    ev_series = - a_ev / (2 * np.arange(1, a_ev_l + 1) - 1) * p_m_ones(a_ev_l)
    odd_series = a_odd / np.arange(1, a_odd_l + 1) * p_m_ones(a_odd_l)
    c = pi / (3 * sum(ev_series))
    z_sh = 1.5 * c ** 2 / pi * sum(odd_series)
    return z_sh, c


def ρ2(a: np.array, z, z_s: float, z_0: float, positive_f=False):
    """
    Gives the Fourier approximation of the nuclear shape, as described
    in the main program block
    """
    n_pi = .5 * pi * np.arange(1, len(a))
    a_ev = a[1::2]
    a_odd = a[2::2]
    u = (z - z_s) / z_0
    if type(z) not in (float, np.float64):
        u_pi_ev = np.array([el * u for el in n_pi[::2]]).T
        u_pi_odd = np.array([el * u for el in n_pi[1::2]]).T
        ρ_2 = (a_ev * np.cos(u_pi_ev)).sum(axis=1)\
            + (a_odd * np.sin(u_pi_odd)).sum(axis=1)
        if positive_f:
            ρ_2[ρ_2 < 0] = 0
            z = z[ρ_2 > 0]
            ρ_2 = ρ_2[ρ_2 > 0]
        return z, ρ_2
    else:
        return z, sum(a_ev * np.cos(n_pi[::2] * u))\
            + sum(a_odd * np.sin(n_pi[1::2] * u))


@nb.njit(fastmath=True, nogil=True)
def ρ2_jit(a, z, z_s, z_0, positive_f):
    """
    Accelerated version of the above proceedure
    """
    n_pi = .5 * pi * np.arange(1, a.size)
    u = (z - z_s) / z_0
    u_pi = np.empty((len(u), len(n_pi)))
    for j, el in enumerate(n_pi):
        u_pi[:, j] = u * el
    ρ_2 = np.sum(a[1::2] * np.cos(u_pi[:, ::2]), axis=1)\
        + np.sum(a[2::2] * np.sin(u_pi[:, 1::2]), axis=1)
    if positive_f:
        z_new, ρ_2_new = [], []
        for i, el in enumerate(ρ_2):
            if el >= 0:
                z_new.append(z[i])
                ρ_2_new.append(ρ_2[i])
        return np.array(z_new), np.array(ρ_2_new)
    return z, ρ_2


def dρ2da_an(a, dim, z, z_s, z_0):
    """
    Calculates first derivative of $ρ^2$(z) with respect to a_n (n > 1)
    """
    if dim < 2:
        return np.zeros_like(z)
    u = (z - z_s) / z_0
    n_pi = .5 * pi * np.arange(1, len(a))
    n_pi = n_pi[dim - 2]
    return np.cos(n_pi * u) if dim % 2 == 0 else np.sin(n_pi * u)


@nb.njit(fastmath=True, nogil=True)
def dρ2da(a, dim, z, z_s, z_0):
    """
    Calculates first derivative of $ρ^2$(z) with respect to a_n (n > 1)
    """
    if dim < 2:
        return np.zeros_like(z)

    a1 = a.copy()
    a1[dim - 1] -= 1e-4
    kw1 = c_z(a1)

    a2 = a.copy()
    a2[dim -1] += 1e-4
    kw2 = c_z(a2)

    return .5e4 * (ρ2_jit(a2, z, *kw2, False)[1] - ρ2_jit(a1, z, *kw1, False)[1])


def dρ2(a, z, z_s, z_0, order=1):
    """
    Gives the Fourier approximation of the nuclear shape, as described
    in the main program block.
    """
    order = 1 if order not in (1, 2) else order
    const = .5 * pi / z_0
    n = np.arange(1, len(a))
    n_pi = .5 * n * pi
    a_ev = a[1::2]
    a_odd = a[2::2]
    u = (z - z_s) / z_0
    if type(z) not in (float, np.float64):
        u_pi_ev = np.array([el * u for el in n_pi[::2]]).T
        u_pi_odd = np.array([el * u for el in n_pi[1::2]]).T
        if order == 2:
            return - const ** 2 * ((a_ev * n[::2]**2 * np.cos(u_pi_ev)).sum(axis=1) +
                             (a_odd * n[1::2]**2 * np.sin(u_pi_odd)).sum(axis=1)
                            )
        return const * ((a_odd * n[1::2] * np.cos(u_pi_odd)).sum(axis=1)
                        -(a_ev * n[::2] * np.sin(u_pi_ev)).sum(axis=1)
                        )
              
    else:
        if order == 2:
            return - const**2 * (sum(a_ev * n[::2]**2 * np.cos(n_pi[::2] * u)) +
                              sum(a_odd * n[1::2]**2 * np.sin(n_pi[1::2] * u))
                             )
        return const * (sum(a_odd * n[1::2] * np.cos(n_pi[1::2] * u))
                        - sum(a_ev * n[::2] * np.sin(n_pi[::2] * u))
                        )


@nb.njit(fastmath=True)
def dρ2_jit(a, z, z_s, z_0, order):
    """
    Gives the Fourier approximation of the nuclear shape, as described
    in the main program block
    """
    order = 1 if order not in (1, 2) else order
    const = .5 * pi / z_0
    n = np.arange(1, len(a))
    n_pi = .5 * n * pi
    a_ev = a[1::2]
    a_odd = a[2::2]
    u = (z - z_s) / z_0

    u_pi = np.empty((len(u), len(n_pi)))
    for j, el in enumerate(n_pi):
        u_pi[:, j] = u * el
    if order == 2:
        return - const ** 2 * ((a_ev * n[::2]**2 * np.cos(u_pi[:, ::2])
                                ).sum(axis=1) +
                               (a_odd * n[1::2]**2 * np.sin(u_pi[:, 1::2])
                                ).sum(axis=1)
                               )
    return const * ((a_odd * n[1::2] * np.cos(u_pi[:, 1::2])).sum(axis=1)
                    - (a_ev * n[::2] * np.sin(u_pi[:, ::2])).sum(axis=1)
                    )

###############################################################################
#     Calculate function g(phi) determining the non-axiality of rho^2(z,phi)
#     and its 1st and 2nd derivatives (see comment about non-axiality 
#     parameter q_1 = eta in subroutine qtoa)
@nb.vectorize
def g(φ, η):
    return (1. - η**2) / (1 + η**2 + 2 * η * np.cos(2*φ))


@nb.vectorize
def dgdφ(φ, η):
    return 4 * η * np.sin(2*φ) * (1. - η**2) / (1 + η**2
                                                + 2 * η * np.cos(2 * φ)) ** 2


@nb.vectorize
def d2gdφ2(φ, η):
    return 8 * (1 - η**2) * η * np.cos(2*φ) / (1 + η**2 +
                                               2 * η * np.cos(2*φ)) ** 2 \
            + 32 * (1 - η**2) * (η * np.sin(2*φ)) ** 2 \
                / (1 + η**2 + 2 * η * np.cos(2*φ)) ** 3


@nb.vectorize
def dgdη(φ, η):
    return - 2 * (2 * η + (1 + η**2)
                   * np.cos(2*φ)) / (1 + η**2 + 2 * η * np.cos(2*φ)) ** 2

###############################################################################
###############################################################################

@nb.njit(fastmath=True, nogil=True)
def bcong(z, ρ_2, a, z_sh, z_0):
    """
       Determines
       bw           deformation dependence of congruence energy
                    (Myers, Swiatecki, Nucl. Phys. A612, 249 (1997));
       bf = Af/A    ratio of lighter-fragment mass to the total mass;
       r12          distance between fragment mass centers;
       rn           neck radius.
    """

    dρ2dz = dρ2_jit(a, z, z_sh, z_0, 1)
    V = np.zeros_like(z)
    cr = np.zeros_like(z)

    V = cum_trapz(ρ_2, z)
    cr = cum_trapz(ρ_2 * z, z)

    vol = .75 * V[-1]

    extr_list = []
    for i, el in enumerate(dρ2dz[:-1]):
        if copysign(1, el) != copysign(1, dρ2dz[i + 1]) and i != 0:
            extr_list.append(i)

    if len(extr_list) == 1:
        bw = 1
        bf = V[extr_list[0]] / V[-1]
        rn = sqrt(ρ_2[extr_list[0]])
        r12 = (cr[-1] - cr[extr_list[0] + 1]) / (V[-1] - V[extr_list[0] + 1])\
                    - cr[extr_list[0]] / V[extr_list[0]]
        z_neck = z[extr_list[0]]
    else:
        bw = 2 - 2 * sqrt(ρ_2[extr_list[1]]) / (sqrt(ρ_2[extr_list[1]])
                                                + sqrt(ρ_2[extr_list[2]]))
        bf = V[extr_list[1]] / V[-1]
        rn = sqrt(ρ_2[extr_list[1]])
        r12 = (cr[-1] - cr[extr_list[1] + 1]) / (V[-1] - V[extr_list[1] + 1])\
                    - cr[extr_list[1]] / V[extr_list[1]]
        z_neck = z[extr_list[1]]
    return bw, round(bf, 5), r12, vol, rn, z_neck


@nb.njit(fastmath=True, nogil=True)
def bcong_Dobr(z, ρ_2, a, z_sh, z_0):
    """
       Determines
       bw           deformation dependence of congruence energy
                    (Myers, Swiatecki, Nucl. Phys. A612, 249 (1997));
       bf = Af/A    ratio of lighter-fragment mass to the total mass;
       r12          distance between fragment mass centers;
       rn           neck radius.
    """

    V = cum_trapz(ρ_2, z)
    vol = .75 * V[-1]

    r_norm = 1
    if 1 - vol > 1e-5:
        r_norm = vol ** (- 1 / 3)
        z_0 *= r_norm
        z_sh *= r_norm
        z = np.linspace(z_sh - z_0, z_0 + z_sh, len(z))
        z, ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)
    else:
        vol = 1

    cr = cum_trapz(ρ_2 * z, z)

    dρ2dz = dρ2_jit(a, z, z_sh, z_0, 1)

    extr_list = []
    for i, el in enumerate(dρ2dz[:-1]):
        if copysign(1, el) != copysign(1, dρ2dz[i + 1]) and i != 0:
            extr_list.append(i)

    if len(extr_list) == 1:
        bw = 1
        i_neck = extr_list[0]
        rn = sqrt(ρ_2[i_neck])
    else:
        i_neck = extr_list[1]
        rn = sqrt(ρ_2[i_neck])
        bw = 2 - 2 * rn / (rn  + sqrt(ρ_2[extr_list[-1]]))
    
    i_neck = i_neck // 2 if i_neck + 1 == len(cr) else i_neck

    bf = V[i_neck] / V[-1]
    r12 = ((cr[-1] - cr[i_neck + 1]) / (V[-1] - V[i_neck + 1])
           - cr[i_neck] / V[i_neck])
    z_neck = z[i_neck]
    
    return bw, bf, r12, vol, rn, z_neck, i_neck, z_0, z_sh


def fcs(q, c, r0, z0, zsh, bw, bf, r12, vol, rn):
    """
    Calculates:
        bs, bk, bc, bw   LD shape functions corresponding respectively to 
                         surface, curvature, Coulomb and congruence energy;
        bf,r12           mass ratio of nascent fission fragments and their
                         center-of-mass distance (see bcong);
        bx,bz            inverse moments of inertia relative to a sphere
                         j_0/j_x, j_0/j_z;
        bq               quadrupole moment in units of e*Z*R_0**2;
        vol              volume of the part with rho^2>0, fission occurs
                         when vol>1;
        rn               neck radius in units of the radius R_0 of the
                         corresponding spherical nucleus.
    """
    global ng1, ng2, ng3
    global xg1, wg1, xg2, wg2, xg3, wg3

    ars = np.zeros(ng1+1)
    adrs = np.zeros(ng1+1)
    addrs = np.zeros(ng1+1)
    ag1 = np.zeros(ng2+1)
    adg1 = np.zeros(ng2+1)
    addg1 = np.zeros(ng2+1)
    ag2 = np.zeros(ng3+1)
    adg2 = np.zeros(ng3+1)
    cph = np.zeros([ng2+1,ng3+1])
    sph = np.zeros([ng2+1,ng3+1])

#  Recall that in the whole program r0 = 1.0
    r_0 = 1
    a = q_to_a(q)
    c, z_sh = c_z(a)
    z_0 = c * r_0
    z = np.linspace(z_sh - z_0, z_0 + z_sh, 400)
    ρ_2 = ρ2(a, z, z_sh, z_0)

    bw, bf, r12, vol, rn = bcong(z, ρ_2)
    if (vol - 1) > 1e-5:
        r_0 = 1 / vol ** (1/3)
        z_0 = c * r_0
        z_sh *= r_0
        z = np.linspace(z_sh - z_0, z_0 + z_sh, 400)
        ρ_2 = ρ2(a, z, z_sh, z_0)

#   Nonaxiallity parameter
    η = a[0]

#   Evaluation of some time consuming functions on the integration nodes
#   phi1 varies from 0 to pi/2

    φ1 = np.linspace(0, .25 * pi)
    φ2 = np.linspace(0, pi)
    Δφ = np.array([φ2 - el for el in φ1])
    ag_1 = np.sqrt(g(φ1, η))
    ag_2 = np.sqrt(g(φ2, η))
    adg1= 0.5 / ag_1 * dgdφ(φ1, η)
    adg2= 0.5 / ag_2 * dgdφ(φ2, η)
    addg1 = 0.5 / ag_1 * (d2gdφ2(φ1, η) - 2 * adg1 ** 2)

    cosΔφ = np.cos(Δφ)
    sinΔφ = np.sin(Δφ)

#   z varies from z_min to z_max
    for i in range(1, ng1 + 1):
        u = xg1[i]
        z = z0 * u + zsh

#   rho2s and its first and second derivative with respect to z
        r2s, r0, c, z0, zsh = rho2s(z, a)
        rs = sqrt(r2s)
        dr2s = drho2s(z, a, 1)
        drs = 0.5 / rs * dr2s
        ddr2s = drho2s(z, a, 2)
        ars = np.sqrt(ρ_2)
        adrs = np.gradient(ars, z)
        addrs = np.gradient(adrs, z)

#   Evaluation of the contributions determining the surface, curvature and
#   Coulomb shape functions bs, bk, bc, quadrupole moment bq and (inverse)
#   rotational moments of inertia bx, bz
    sur = 0
    cur = 0
    cou = 0
    rotx = 0
    rotz = 0
    qua = 0

#   integration over z1
    for i in range(1, ng1 + 1):
        u1 = xg1[i]
        z1 = z0 * u1 + zsh
        a1 = wg1[i]
        rs1 = ars[i]
        if rs1 == 0:
            continue
        drs1 = adrs[i]
        ddrs1 = addrs[i]

#   integration over phi1 from 0 to pi/2
        for j in range(1, ng2 + 1):
            u2 = xg2[j]
            ph1 = 0.25 * pi * (u2 + 1)
            a2 = wg2[j]
            g1 = ag1[j]
            dg1 = adg1[j]
            ddg1 = addg1[j]
            r1 = rs1 * g1
            dr1dz = drs1 * g1
            dr1dp = rs1 * dg1
            d2r1dz2 = ddrs1 * g1
            d2r1dzdp = drs1 * dg1
            d2r1dp2 = rs1 * ddg1
            sur=sur+a1*a2*sqrt(r1**2+dr1dp**2+(r1*dr1dz)**2)
            cur=cur+a1*a2*((r1-d2r1dp2)*r1*dr1dz**2 +r1**2-r1*d2r1dp2 \
                +2*dr1dp**2+2*dr1dz*dr1dp*r1*d2r1dzdp-r1**3*d2r1dz2 \
                           -r1*d2r1dz2*dr1dp**2)/(r1**2+dr1dp**2+(r1*dr1dz)**2)
            rotx=rotx+a1*a2*(0.5*r1**4*(math.sin(ph1))**2+r1**2*z1**2)
            rotz=rotz+a1*a2*0.5*r1**4
            qua=qua+a1*a2*(z1**2*r1**2-0.25*r1**4)

#   integration over phi2 from 0 to 2pi
            for k in range(1,ng3+1):
                ph2=pi*(xg3[k]+1)
                a3=wg3[k]
                g2=ag2[k]
                dg2=adg2[k]
                cph12=cph[j,k]
                sph12=sph[j,k]

#   integration over z2
                for l in range(1,ng1+1):
                    z2=z0*xg1[l]+zsh
                    a4=wg1[l]
                    rs2=ars[l]
                    drs2=adrs[l]
                    r2=rs2*g2
                    dr2dz=drs2*g2
                    dr2dp=rs2*dg2
                    cou=cou+a1*a2*a3*a4*(r1**2-r1*r2*cph12-r2*sph12*dr1dp \
                    -r1*(z1-z2)*dr1dz)*(r2**2-r2*r1*cph12+r1*sph12*dr2dp \
                    +r2*(z1-z2)*dr2dz)/sqrt(r1**2+r2**2-2.*r1*r2*cph12+(z1-z2)**2)

#    print(sur,cou)
    bs = 0.25 * z0 / r0 ** 2 * sur
    bk = 0.125 * z0 / r0 * cur
    bx = 16 / 15 * r0**5 / (z0 * rotx)
    bz = 16 / 15 * r0**5 / (z0 * rotz)
    bq = 0.75 * z0 / r0**3 * qua
    bc = 5 / 64 * z0**2 / r0**5 * cou

    return bs, bk, bx, bz, bq, bc


@nb.njit(fastmath=True, nogil=True)
def fcs_pythonic(q):
    """
    Calculates:
        bs, bk, bc, bw   LD shape functions corresponding respectively to 
                         surface, curvature, Coulomb and congruence energy;
        bf,r12           mass ratio of nascent fission fragments and their
                         center-of-mass distance (see bcong);
        bx,bz            inverse moments of inertia relative to a sphere
                         j_0/j_x, j_0/j_z;
        bq               quadrupole moment in units of e*Z*R_0**2;
        vol              volume of the part with rho^2>0, fission occurs
                         when vol>1;
        rn               neck radius in units of the radius R_0 of the
                         corresponding spherical nucleus.
    """

    N = 500

#  Recall that in the whole program r0 = 1.0
    r0 = 1
    a = q_to_a(q)
    z_sh, c = c_z(a)
    z_0 = c * r0
    z = np.linspace(z_sh - z_0, z_0 + z_sh, N)

    z, ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)

    bw, bf, r12, vol, rn, _ = bcong(z, ρ_2, a, z_sh, z_0)
    if (vol - 1) > 1e-5:
        r_0 = 1 / vol ** (1/3)
        z_0 = c * r_0
        z_sh *= r_0
        z = np.linspace(z_sh - z_0, z_0 + z_sh, N)
        ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)


#   Nonaxiallity parameter
    η = a[0]

#   Evaluation of some time consuming functions on the integration nodes
#   phi1 varies from 0 to pi/2

    ag_1 = np.sqrt(g(φ1, η))
    adg1= 0.5 / ag_1 * dgdφ(φ1, η)
    addg1 = 0.5 / ag_1 * (d2gdφ2(φ1, η) - 2 * adg1 ** 2)

    ag_2 = np.sqrt(g(φ2, η))
    adg2= 0.5 / ag_2 * dgdφ(φ2, η)

#   z varies from z_min to z_max
    z = z_0 * xg1 + z_sh

#   rho2s and its first and second derivative with respect to z
    r2s = ρ2_jit(a, z, z_sh, z_0, False)[1]

    rs = np.sqrt(np.maximum(1e-10 * np.ones_like(z), r2s))

    dr2s = dρ2_jit(a, z, z_sh, c, 1)

    drs = 0.5 / rs * dr2s
    ddr2s = dρ2_jit(a, z, z_sh, c, 2)
    ddrs = 0.5 / rs * (ddr2s - 2 * drs ** 2)

#   Evaluation of the contributions determining the surface, curvature and
#   Coulomb shape functions bs, bk, bc, quadrupole moment bq and (inverse)
#   rotational moments of inertia bx, bz
    sur = 0
    cur = 0
    cou = 0
    rotx = 0
    rotz = 0
    qua = 0

#   integration over z1
    for z1, a1, rs1, drs1, ddrs1 in zip(z, wg1, rs, drs, ddrs):
#   integration over phi1 from 0 to pi/2
        for φ_1, a2, g1, dg1, ddg1, j in zip(φ1, wg2, ag_1, adg1, addg1,
                                             range(ng2)):
            r1 = rs1 * g1
            r12 = r1**2
            dr1dz = drs1 * g1
            dr1dp = rs1 * dg1
            dr1dp2 = dr1dp**2

            d2r1dz2 = ddrs1 * g1
            d2r1dzdp = drs1 * dg1
            d2r1dp2 = rs1 * ddg1
            sur += a1 * a2 * sqrt(r12 + dr1dp2 + (r1 * dr1dz)**2)
            cur += a1 * a2 * ((r1 - d2r1dp2) * r1 * dr1dz**2 + r12
                              - r1 * d2r1dp2 + 2 * dr1dp2
                              + 2 * dr1dz * dr1dp * r1 * d2r1dzdp
                              - r1 ** 3 * d2r1dz2 -r1 * d2r1dz2 * dr1dp2
                              ) / (r12 + dr1dp2 + (r1 * dr1dz)**2)
            rotx += a1 * a2 * (.5 * r1**4 * sin(φ_1)**2 + r12 * z1**2)
            rotz += a1 * a2 * .5* r1**4
            qua += a1 * a2 * (z1**2 * r1**2 - .25 * r1**4)

#   integration over phi2 from 0 to 2pi
            for φ_2, a3, g2, dg2, cph12, sph12 in zip(φ2, wg3, ag_2, adg2,
                                                      cosΔφ[j], sinΔφ[j]):

#   integration over z2
                for z2, a4, rs2, drs2 in zip(z, wg1, rs, drs):
                    Δz = z1 - z2
                    r2 = rs2*g2
                    r22 = r2**2
                    r1r2 = r1 * r2 * cph12
                    dr2dz = drs2*g2
                    dr2dp = rs2*dg2
                    denumenator = sqrt(r12 + r22 - 2 * r1r2 + Δz**2)
                    numenator = (r1**2 - r1 * r2 * cph12 - r2 * sph12 * dr1dp 
                                 - r1 * (z1 - z2) * dr1dz)\
                        * (r22 - r1r2 + r1 * sph12 * dr2dp  + r2 * Δz * dr2dz)
                    cou += a1 * a2 * a3 * a4 * numenator/ denumenator

#    print(sur,cou)
    
    b_surf = 0.25 * z_0 * sur
    b_curv = 0.125 * z_0 * cur
    J_x = 16 / 15 / (z_0 * rotx)
    J_z = 16 / 15 / (z_0 * rotz)
    bq = 0.75 * z_0 * qua
    b_coul = 5 / 64 * z_0**2 * cou

    return b_surf, b_curv, J_x, J_z, bq, b_coul


@nb.njit(fastmath=True, nogil=True)
def fcs_short(q):
    """
    Calculates:
        bs, bk, bc, bw   LD shape functions corresponding respectively to 
                         surface, curvature, Coulomb and congruence energy;
        bf,r12           mass ratio of nascent fission fragments and their
                         center-of-mass distance (see bcong);
        bx,bz            inverse moments of inertia relative to a sphere
                         j_0/j_x, j_0/j_z;
        bq               quadrupole moment in units of e*Z*R_0**2;
        vol              volume of the part with rho^2>0, fission occurs
                         when vol>1;
        rn               neck radius in units of the radius R_0 of the
                         corresponding spherical nucleus.
    """

    N = 200

#  Recall that in the whole program r0 = 1.0
    r0 = 1
    a = q_to_a(q)
    z_sh, c = c_z(a)
    z_0 = c * r0
    z = np.linspace(z_sh - z_0, z_0 + z_sh, N)

    z, ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)

    bw, bf, r12, vol, rn, z_neck, z_neck_ind, z_0, z_sh = bcong_Dobr(z, ρ_2, a,
                                                                     z_sh, z_0)
    # bw, bf, r12, vol, rn, _ = bcong_Dobr(z, ρ_2, a, z_sh, z_0)
    # if (vol - 1) > 1e-5:
    #     r_0 = 1 / vol ** (1/3)
    #     z_0 = c * r_0
    #     z_sh *= r_0
    #     z = np.linspace(z_sh - z_0, z_0 + z_sh, N)
    #     ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)

#   Nonaxiallity parameter
    η = a[0]

#   Evaluation of some time consuming functions on the integration nodes
#   phi1 varies from 0 to pi/2

    ag_1 = np.sqrt(g(φ1, η))
    adg1= 0.5 / ag_1 * dgdφ(φ1, η)
    addg1 = 0.5 / ag_1 * (d2gdφ2(φ1, η) - 2 * adg1 ** 2)
    ag_2 = np.sqrt(g(φ2, η))
    adg2= 0.5 / ag_2 * dgdφ(φ2, η)

#   z varies from z_min to z_max
    z = z_0 * xg1 + z_sh

#   rho2s and its first and second derivative with respect to z
    r2s = ρ2_jit(a, z, z_sh, z_0, False)[1]

    rs = np.sqrt(np.maximum(1e-10 * np.ones_like(z), r2s))

    dr2s = dρ2_jit(a, z, z_sh, c, 1)

    drs = 0.5 / rs * dr2s
    ddr2s = dρ2_jit(a, z, z_sh, c, 2)
    ddrs = 0.5 / rs * (ddr2s - 2 * drs ** 2)

#   Evaluation of the contributions determining the surface, curvature and
#   Coulomb shape functions bs, bk, bc, quadrupole moment bq and (inverse)
#   rotational moments of inertia bx, bz
    sur = 0
    cur = 0
    cou = 0

#   integration over z1
    for z1, a1, rs1, drs1, ddrs1 in zip(z, wg1, rs, drs, ddrs):
#   integration over phi1 from 0 to pi/2
        for φ_1, a2, g1, dg1, ddg1, j in zip(φ1, wg2, ag_1, adg1, addg1,
                                             range(ng2)):
            r1 = rs1 * g1
            r12 = r1**2
            dr1dz = drs1 * g1
            dr1dp = rs1 * dg1
            dr1dp2 = dr1dp**2

            d2r1dz2 = ddrs1 * g1
            d2r1dzdp = drs1 * dg1
            d2r1dp2 = rs1 * ddg1
            sur += a1 * a2 * sqrt(r12 + dr1dp2 + (r1 * dr1dz)**2)
            cur += a1 * a2 * ((r1 - d2r1dp2) * r1 * dr1dz**2 + r12
                              - r1 * d2r1dp2 + 2 * dr1dp2
                              + 2 * dr1dz * dr1dp * r1 * d2r1dzdp
                              - r1 ** 3 * d2r1dz2 -r1 * d2r1dz2 * dr1dp2
                              ) / (r12 + dr1dp2 + (r1 * dr1dz)**2)

#   integration over phi2 from 0 to 2pi
            for φ_2, a3, g2, dg2, cph12, sph12 in zip(φ2, wg3, ag_2, adg2,
                                                      cosΔφ[j], sinΔφ[j]):

#   integration over z2
                for z2, a4, rs2, drs2 in zip(z, wg1, rs, drs):
                    Δz = z1 - z2
                    r2 = rs2*g2
                    r22 = r2**2
                    r1r2 = r1 * r2 * cph12
                    dr2dz = drs2*g2
                    dr2dp = rs2*dg2
                    denumenator = sqrt(r12 + r22 - 2 * r1r2 + Δz**2)
                    numenator = (r1**2 - r1 * r2 * cph12 - r2 * sph12 * dr1dp 
                                 - r1 * (z1 - z2) * dr1dz)\
                        * (r22 - r1r2 + r1 * sph12 * dr2dp  + r2 * Δz * dr2dz)
                    cou += a1 * a2 * a3 * a4 * numenator/ denumenator

#    print(sur,cou)
    b_surf = 0.25 * z_0 * sur
    b_curv = 0.125 * z_0 * cur
    b_coul = 5 / 64 * z_0**2 * cou

    return b_coul, b_surf, b_curv


@nb.njit(fastmath=True, nogil=True)
def BCoul_pythonic(q):
    """
    Calculates:
        bs, bk, bc, bw   LD shape functions corresponding respectively to 
                         surface, curvature, Coulomb and congruence energy;
        bf, r1_2           mass ratio of nascent fission fragments and their
                         center-of-mass distance (see bcong);
        bx,bz            inverse moments of inertia relative to a sphere
                         j_0/j_x, j_0/j_z;
        bq               quadrupole moment in units of e*Z*R_0**2;
        vol              volume of the part with rho^2>0, fission occurs
                         when vol>1;
        rn               neck radius in units of the radius R_0 of the
                         corresponding spherical nucleus.
    """

#  Recall that in the whole program r0 = 1.0
    r0 = 1
    a = q_to_a(q)
    z_sh, c = c_z(a)
    z_0 = c * r0
    z = np.linspace(z_sh - z_0, z_0 + z_sh, 400)

    z, ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)

    bw, bf, r1_2, vol, rn, z_nck = bcong(z, ρ_2, a, z_sh, z_0)
    if abs(vol - 1) > 1e-5:
        r_0 = 1 / vol ** (1/3)
        z_0 = c * r_0
        z_sh *= r_0
        z = np.linspace(z_sh - z_0, z_0 + z_sh, 400)
        z, ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)

    lim1, lim2 = .5 * (z_nck + z_0 - z_sh), .5 * (- z_nck + z_0 + z_sh)
    shift1, shift2 = .5 * (z_nck - z_0 + z_sh), .5 * (z_nck + z_0 + z_sh)

#   Nonaxiallity parameter
    η = a[0]

#   Evaluation of some time consuming functions on the integration nodes
#   phi1 varies from 0 to pi/2

    ag_1 = np.sqrt(g(φ1, η))
    adg1= 0.5 / ag_1 * dgdφ(φ1, η)
    addg1 = 0.5 / ag_1 * (d2gdφ2(φ1, η) - 2 * adg1 ** 2)
    ag_2 = np.sqrt(g(φ2, η))
    adg2= 0.5 / ag_2 * dgdφ(φ2, η)

#   z varies from z_min to z_max
    z, z1, z2 = (z_0 * xg1 + z_sh, 
                 xg1 * lim1 + shift1,
                 xg1 * lim2 + shift2)

#   rho2s and its first and second derivative with respect to z
    ρ2s, ρ2s1, ρ2s2 = (ρ2_jit(a, z, z_sh, z_0, False)[1],
                       ρ2_jit(a, z1, z_sh, z_0, False)[1],
                       ρ2_jit(a, z2, z_sh, z_0, False)[1])

    ρs = np.sqrt(np.maximum(1e-10 * np.ones_like(z), ρ2s))
    ρs1 = np.sqrt(np.maximum(1e-10 * np.ones_like(z1), ρ2s1))
    ρs2 = np.sqrt(np.maximum(1e-10 * np.ones_like(z1), ρ2s2))

    dρ2s, dρ2s1, dρ2s2 = (dρ2_jit(a, z, z_sh, c, 1),
                          dρ2_jit(a, z1, z_sh, c, 1),
                          dρ2_jit(a, z2, z_sh, c, 1))
    dρs, dρs1, dρs2 = .5 / ρs * dρ2s, .5 / ρs1 * dρ2s1, .5 / ρs2 * dρ2s2

    d2ρ2s, d2ρ2s1, d2ρ2s2 = (dρ2_jit(a, z, z_sh, c, 2),
                             dρ2_jit(a, z1, z_sh, c, 2),
                             dρ2_jit(a, z2, z_sh, c, 2))

    d2ρs, d2ρs1, d2ρs2 = (.5 / ρs * (d2ρ2s - 2 * dρs**2),
                          .5 / ρs1 * (d2ρ2s1 - 2 * dρs1**2),
                          .5 / ρs2 * (d2ρ2s2 - 2 * dρs2**2))

#   Evaluation of the contributions determining the surface, curvature and
#   Coulomb shape functions bs, bk, bc, quadrupole moment bq and (inverse)
#   rotational moments of inertia bx, bz

    cou = 0
    cou1 = 0
    cou2 = 0

    loop_val1 = (z, z1, z2, wg1, ρs, ρs1, ρs2, dρs, dρs1, dρs2, d2ρs, d2ρs1,
                 d2ρs2)

    loop_val2 = (z, z1, z2, wg1, ρs, ρs1, ρs2, dρs, dρs1, dρs2)

#   integration over z1
    for (z01, z11, z21, a1, ρ01, ρ11, ρ21, dρ01, dρ11, dρ21, ddρ01, ddρ11,
         ddρ21) in zip(*loop_val1):
#   integration over phi1 from 0 to pi/2
        for φ_1, a2, g1, dg1, ddg1, j in zip(φ1, wg2, ag_1, adg1, addg1,
                                             range(ng2)):
            r01, r11, r21 = ρ01 * g1, ρ11 * g1, ρ21 * g1
            r01_2, r11_2, r21_2 = r01 ** 2, r11 ** 2, r21 ** 2
            dr01dz, dr11dz, dr21dz = dρ01 * g1, dρ11 * g1, dρ21 * g1

            dr01dφ, dr11dφ, dr21dφ = ρ01 * dg1, ρ11 * dg1, ρ21 * dg1

#   integration over phi2 from 0 to 2pi
            for φ_2, a3, g2, dg2, cph12, sph12 in zip(φ2, wg3, ag_2, adg2,
                                                      cosΔφ[j], sinΔφ[j]):

#   integration over z2
                for (z02, z12, z22, a4, ρ02, ρ12, ρ22, dρ02, dρ12, dρ22)\
                    in zip(*loop_val2):
                    f = a1 * a2 * a3 * a4
                    # f *= a3 * a4
                    Δz0, Δz1, Δz2 = z01 - z02, z11 - z12, z21 - z22

                    r02, r12, r22 = ρ02 * g2, ρ12 * g2, ρ22 * g2
                    r02_2, r12_2, r22_2 = r02 ** 2, r12 ** 2, r22 ** 2

                    r01r02, r11r12, r21r22 = (r01 * r02 * cph12,
                                              r11 * r12 * cph12,
                                              r21 * r22 * cph12)

                    dr02dz, dr12dz, dr22dz = dρ02 * g2, dρ12 * g2, dρ22 * g2
                    dr02dφ , dr12dφ, dr22dφ = ρ02 * dg2, ρ12 * dg2, ρ22 * dg2

                    denumenator = sqrt(r01_2 + r02_2 - 2 * r01r02 + Δz0 ** 2)
                    numenator = (r01_2 - r01r02 - r02 * sph12 * dr01dφ
                                 - r01 * Δz0 * dr01dz)\
                                * (r02_2 - r01r02 + r01 * sph12 * dr02dφ
                                   + r02 * Δz0 * dr02dz)

                    denumenator1 = sqrt(r11_2 + r12_2 - 2 * r11r12 + Δz1 ** 2)
                    numenator1 = (r11_2 - r11r12 - r12 * sph12 * dr11dφ
                                  - r11 * Δz1 * dr11dz)\
                                 * (r12_2 - r11r12 + r11 * sph12 * dr12dφ
                                    + r12 * Δz1 * dr12dz)

                    denumenator2 = sqrt(r21_2 + r22_2 - 2 * r21r22 + Δz2 ** 2)
                    numenator2 = (r21_2 - r21r22 - r22 * sph12 * dr21dφ
                                  - r21 * Δz2 * dr21dz)\
                                 * (r22_2 - r21r22 + r21 * sph12 * dr22dφ
                                    + r22 * Δz2 * dr22dz)

                    cou += f * numenator / denumenator
                    cou1 += f * numenator1 / denumenator1
                    cou2 += f * numenator2 / denumenator2

    bc = 5 / (64 * r0 ** 5) * z_0 ** 2 * cou
    bc1 = 5 / 64 * bf ** ( - 5 / 3) * lim1 ** 2 * cou1
    bc2 = 5 / 64 * (1 - bf) ** (- 5 / 3) * lim2 ** 2 * cou2

    return np.array([bc, bc1, bc2])


@nb.njit(fastmath=True, nogil=True)
def surface_coefficients(q):
    """
    Calculates:
        bs, bk, bc, bw   LD shape functions corresponding respectively to 
                         surface, curvature, Coulomb and congruence energy;
        bf,r12           mass ratio of nascent fission fragments and their
                         center-of-mass distance (see bcong);
        bx,bz            inverse moments of inertia relative to a sphere
                         j_0/j_x, j_0/j_z;
        bq               quadrupole moment in units of e*Z*R_0**2;
        vol              volume of the part with rho^2>0, fission occurs
                         when vol>1;
        rn               neck radius in units of the radius R_0 of the
                         corresponding spherical nucleus.
    """

    N = 1000

#  Recall that in the whole program r0 = 1.0
    r0 = 1
    a = q_to_a(q)
    z_sh, c = c_z(a)
    z_0 = c * r0
    z = np.linspace(z_sh - z_0, z_0 + z_sh, N)

    z, ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)

    bw, bf, r12, vol, rn, z_nck, _, z_0, z_sh = bcong_Dobr(z, ρ_2, a,
                                                           z_sh, z_0)
    
    if (vol - 1) > 1e-5:
        z = np.linspace(z_sh - z_0, z_0 + z_sh, N)
        z, ρ_2 = ρ2_jit(a, z, z_sh, z_0, True)

    lim1, lim2 = .5 * (z_nck + z_0 - z_sh), .5 * (- z_nck + z_0 + z_sh)
    shift1, shift2 = .5 * (z_nck - z_0 + z_sh), .5 * (z_nck + z_0 + z_sh)

#   Nonaxiallity parameter
    η = a[0]

#   Evaluation of some time consuming functions on the integration nodes
#   phi1 varies from 0 to pi/2

    ag_1 = np.sqrt(g(φ1, η))
    adg1= 0.5 / ag_1 * dgdφ(φ1, η)
    addg1 = 0.5 / ag_1 * (d2gdφ2(φ1, η) - 2 * adg1 ** 2)
    ag_2 = np.sqrt(g(φ2, η))
    adg2= 0.5 / ag_2 * dgdφ(φ2, η)

#   z varies from z_min to z_max
    z, z1, z2 = (z_0 * xg1 + z_sh, 
                 xg1 * lim1 + shift1,
                 xg1 * lim2 + shift2)

#   rho2s and its first and second derivative with respect to z
    ρ2s, ρ2s1, ρ2s2 = (ρ2_jit(a, z, z_sh, z_0, False)[1],
                       ρ2_jit(a, z1, z_sh, z_0, False)[1],
                       ρ2_jit(a, z2, z_sh, z_0, False)[1])

    ρs = np.sqrt(np.maximum(1e-10 * np.ones_like(z), ρ2s))
    ρs1 = np.sqrt(np.maximum(1e-10 * np.ones_like(z1), ρ2s1))
    ρs2 = np.sqrt(np.maximum(1e-10 * np.ones_like(z2), ρ2s2))

    dρ2s, dρ2s1, dρ2s2 = (dρ2_jit(a, z, z_sh, z_0, 1),
                          dρ2_jit(a, z1, z_sh, z_0, 1),
                          dρ2_jit(a, z2, z_sh, z_0, 1))
    dρs, dρs1, dρs2 = .5 / ρs * dρ2s, .5 / ρs1 * dρ2s1, .5 / ρs2 * dρ2s2

    d2ρ2s, d2ρ2s1, d2ρ2s2 = (dρ2_jit(a, z, z_sh, z_0, 2),
                             dρ2_jit(a, z1, z_sh, z_0, 2),
                             dρ2_jit(a, z2, z_sh, z_0, 2))

    d2ρs, d2ρs1, d2ρs2 = (.5 / ρs * (d2ρ2s - 2 * dρs**2),
                          .5 / ρs1 * (d2ρ2s1 - 2 * dρs1**2),
                          .5 / ρs2 * (d2ρ2s2 - 2 * dρs2**2))

#   Evaluation of the contributions determining the surface, curvature and
#   Coulomb shape functions bs, bk, bc, quadrupole moment bq and (inverse)
#   rotational moments of inertia bx, bz

    cou = 0
    cou1 = 0
    cou2 = 0

    sur = 0
    sur1 = 0
    sur2 = 0

    cur = 0
    cur1 = 0
    cur2 = 0

    loop_val1 = (z, z1, z2, wg1, ρs, ρs1, ρs2, dρs, dρs1, dρs2, d2ρs, d2ρs1,
                 d2ρs2)

    loop_val2 = (z, z1, z2, wg1, ρs, ρs1, ρs2, dρs, dρs1, dρs2)

#   integration over z1
    for (z01, z11, z21, a1, ρ01, ρ11, ρ21, dρ01, dρ11, dρ21, ddρ01, ddρ11,
         ddρ21) in zip(*loop_val1):
#   integration over phi1 from 0 to pi/2
        for φ_1, a2, g1, dg1, ddg1, j in zip(φ1, wg2, ag_1, adg1, addg1,
                                             range(ng2)):
            r01, r11, r21 = ρ01 * g1, ρ11 * g1, ρ21 * g1
            r01_2, r11_2, r21_2 = r01 ** 2, r11 ** 2, r21 ** 2
            dr01dz, dr11dz, dr21dz = dρ01 * g1, dρ11 * g1, dρ21 * g1
            d2r01dz2, d2r11dz2, d2r21dz2 = ddρ01 * g1, ddρ11 * g1, ddρ21 * g1

            dr01dφ, dr11dφ, dr21dφ = ρ01 * dg1, ρ11 * dg1, ρ21 * dg1
            dr01dφ_2, dr11dφ_2, dr21dφ_2 = dr01dφ**2 , dr11dφ**2, dr21dφ**2
            d2r01dφ2, d2r11dφ2, d2r21dφ2 = ρ01 * ddg1, ρ11 * ddg1, ρ21 * ddg1
            d2r01dzdφ, d2r11dzdφ, d2r21dzdφ = (dr01dz * dg1, dr11dz * dg1,
                                               dr21dz * dg1)

            f = a1 * a2
            num = r01_2 * (1 + dr01dz**2) + dr01dφ_2
            sur += f * sqrt(num)
            cur += f / num * ((r01 - d2r01dφ2) * r01 * dr01dz**2 + r01_2
                              - r01 * d2r01dφ2 + 2 * dr01dφ_2
                              + 2 * dr01dz * dr01dφ * r01 * d2r01dzdφ
                              - r01 * d2r01dz2 * (r01_2 + dr01dφ_2))

            num1 = r11_2 * (1 + dr11dz**2) + dr11dφ_2
            sur1 += f * sqrt(num1)
            cur1 += f / num1 * ((r11 - d2r11dφ2) * r11 * dr11dz**2 + r11_2
                              - r11 * d2r11dφ2 + 2 * dr11dφ_2
                              + 2 * dr11dz * dr11dφ * r11 * d2r11dzdφ
                              - r11 * d2r11dz2 * (r11_2 + dr11dφ_2))

            num2 = r21_2 * (1 + dr21dz**2) + dr21dφ_2
            sur2 += f * sqrt(num2)
            cur2 += f / num2 * ((r21 - d2r21dφ2) * r21 * dr21dz**2 + r21_2
                              - r21 * d2r21dφ2 + 2 * dr21dφ_2
                              + 2 * dr21dz * dr21dφ * r21 * d2r21dzdφ
                              - r21 * d2r21dz2 * (r21_2 + dr21dφ_2))

#   integration over phi2 from 0 to 2pi
            for φ_2, a3, g2, dg2, cph12, sph12 in zip(φ2, wg3, ag_2, adg2,
                                                      cosΔφ[j], sinΔφ[j]):

#   integration over z2
                for (z02, z12, z22, a4, ρ02, ρ12, ρ22, dρ02, dρ12, dρ22)\
                    in zip(*loop_val2):
                    f = a1 * a2 * a3 * a4
                    # f *= a3 * a4
                    Δz0, Δz1, Δz2 = z01 - z02, z11 - z12, z21 - z22

                    r02, r12, r22 = ρ02 * g2, ρ12 * g2, ρ22 * g2
                    r02_2, r12_2, r22_2 = r02 ** 2, r12 ** 2, r22 ** 2

                    r01r02, r11r12, r21r22 = (r01 * r02 * cph12,
                                              r11 * r12 * cph12,
                                              r21 * r22 * cph12)

                    dr02dz, dr12dz, dr22dz = dρ02 * g2, dρ12 * g2, dρ22 * g2
                    dr02dφ , dr12dφ, dr22dφ = ρ02 * dg2, ρ12 * dg2, ρ22 * dg2

                    denumenator = sqrt(r01_2 + r02_2 - 2 * r01r02 + Δz0 ** 2)
                    numenator = (r01_2 - r01r02 - r02 * sph12 * dr01dφ
                                 - r01 * Δz0 * dr01dz)\
                                * (r02_2 - r01r02 + r01 * sph12 * dr02dφ
                                   + r02 * Δz0 * dr02dz)

                    denumenator1 = sqrt(r11_2 + r12_2 - 2 * r11r12 + Δz1 ** 2)
                    numenator1 = (r11_2 - r11r12 - r12 * sph12 * dr11dφ
                                  - r11 * Δz1 * dr11dz)\
                                 * (r12_2 - r11r12 + r11 * sph12 * dr12dφ
                                    + r12 * Δz1 * dr12dz)

                    denumenator2 = sqrt(r21_2 + r22_2 - 2 * r21r22 + Δz2 ** 2)
                    numenator2 = (r21_2 - r21r22 - r22 * sph12 * dr21dφ
                                  - r21 * Δz2 * dr21dz)\
                                 * (r22_2 - r21r22 + r21 * sph12 * dr22dφ
                                    + r22 * Δz2 * dr22dz)

                    cou += f * numenator / denumenator
                    cou1 += f * numenator1 / denumenator1
                    cou2 += f * numenator2 / denumenator2

    bc = 5 / (64 * r0**5) * z_0 ** 2 * cou
    bs = .25 * z_0 / r0**2 * sur
    bk = .125 * z_0 / r0 * cur

    bc1 = 5 / 64 * bf**(-5 / 3) * lim1**2 * cou1
    bs1 = .25 * lim1 * bf**(-2 / 3) * sur1
    bk1 = .125 * lim1 * bf**(-1 / 3) * cur1

    bc2 = 5 / 64 * (1 - bf)**(-5 / 3) * lim2**2 * cou2
    bs2 = .25 * lim2 * (1 - bf)**(-2 / 3) * sur2
    bk2 = .125 * lim2 * (1 - bf)**(-1 / 3) * cur2

    return bc, bs, bk, bc1, bs1, bk1, bc2, bs2, bk2
    # return bc, bs, bk, bc1, bs1, bk1, bc2, bs2, bk2, bf

if __name__ == "__main__":
    pass
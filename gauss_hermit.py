"""
ПРАВИЛЬНАЯ архитектура Gauss-Hermit интерполяции
Один модуль, оптимальная производительность, чистая архитектура
"""

import numpy as np
import numba as nb
from math import sqrt, exp
from config import DIM as dim, NODES as nodes, H_NODES as h_nodes, GAMMA as γ

###############################################################################
###################### ЕДИНАЯ СИСТЕМА ИНТЕРПОЛЯЦИИ ###########################
###############################################################################

@nb.njit(nb.float64[:](nb.float64[:], nb.int64), fastmath=True)
def round_njt(x, decimals):
    """JIT-оптимизированная версия np.round"""
    out = np.empty(x.shape[0])
    return np.round_(x, decimals, out)


@nb.njit(fastmath=True, inline='always')
def gaussian_weight(u2):
    """Вычисление весовой функции Гаусса-Эрмита"""
    return exp(-u2) * (1.875 - 2.5 * u2 + 0.5 * u2 * u2)


@nb.njit(fastmath=True, nogil=True)
def compute_weights(q, qlim, dq, crd_int):
    """
    Вычисление весовых коэффициентов (выделено для переиспользования)
    """
    crd = (q - qlim[0]) / dq
    f = np.empty((dim, nodes))
    
    for i in range(dim):
        f_sum = 0.0
        for j in range(nodes):
            u2 = (γ * (crd[i] - (crd_int[i] + j - h_nodes))) ** 2
            f[i, j] = gaussian_weight(u2)
            f_sum += f[i, j]
        
        # Нормализация
        inv_sum = 1.0 / f_sum
        for j in range(nodes):
            f[i, j] *= inv_sum
    
    return f


@nb.njit(fastmath=True, nogil=True, cache=True)
def gh_interpolate_scalar(q, qlim, dq, N_q, matrix):
    """
    БАЗОВАЯ скалярная интерполяция - основа для всех остальных
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(nb.intp)
    f = compute_weights(q, qlim, dq, crd_int)

    element = 0.0
    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        fi = f[0, i]
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            fj = fi * f[1, j]
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                element += fj * f[2, k] * matrix[ii, jj, kk]
    
    return element


@nb.njit(fastmath=True, nogil=True, cache=True)
def gh_interpolate_tensor(q, qlim, dq, N_q, tensor):
    """
    Тензорная интерполяция - расширение базовой функции
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(nb.intp)
    f = compute_weights(q, qlim, dq, crd_int)

    result = np.zeros((dim, dim))
    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        fi = f[0, i]
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            fj = fi * f[1, j]
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                fk = fj * f[2, k]
                
                # Векторизованное обновление тензора
                for di in range(dim):
                    for dj in range(dim):
                        result[di, dj] += fk * tensor[ii, jj, kk, di, dj]
    
    return result


@nb.njit(fastmath=True, nogil=True, cache=True)
def gh_interpolate_multi_optimized(q, qlim, dq, N_q, 
                                   scalar_arrays, vector_arrays, tensor_arrays):
    """
    МАКСИМАЛЬНО ОПТИМИЗИРОВАННАЯ множественная интерполяция
    
    ✅ ОДИН цикл для всех величин
    ✅ Векторизованные операции
    ✅ Минимум вызовов функций
    ✅ Оптимальное использование кэша
    
    Parameters:
    -----------
    scalar_arrays : tuple of arrays
        (ar_Vmac, ar_Vmic, ar_den, ...)
    vector_arrays : tuple of arrays  
        (ar_d_Vmac, ar_d_Vmic, ar_d_den, ...)
    tensor_arrays : tuple of arrays
        (ar_invM, ar_G, ar_sqrtG, ar_d_invM, ...)
    """
    crd = (q - qlim[0]) / dq
    crd_int = round_njt(crd, 0).astype(nb.intp)
    f = compute_weights(q, qlim, dq, crd_int)

    # Распаковка входных массивов
    ar_Vmac, ar_Vmic, ar_den = scalar_arrays
    ar_d_Vmac, ar_d_Vmic, ar_d_den = vector_arrays
    ar_invM, ar_G, ar_sqrtG, ar_d_invM = tensor_arrays

    # Инициализация результатов
    scalars = np.zeros(3)  # [Vmac, Vmic, den]
    vectors = np.zeros((3, dim))  # [dVmac, dVmic, d_den]
    tensors = np.zeros((4, dim, dim))  # [invM, G, sqrtG, ...] 
    tensor_3d = np.zeros((dim, dim, dim))  # d_invM

    # ⚡ ЕДИНЫЙ ОПТИМИЗИРОВАННЫЙ ЦИКЛ ⚡
    for i in range(nodes):
        ii = max(0, min(N_q[0] - 1, crd_int[0] + i - h_nodes))
        fi = f[0, i]
        for j in range(nodes):
            jj = max(0, min(N_q[1] - 1, crd_int[1] + j - h_nodes))
            fj = fi * f[1, j]
            for k in range(nodes):
                kk = max(0, min(N_q[2] - 1, crd_int[2] + k - h_nodes))
                weight = fj * f[2, k]

                # Скалярная интерполяция (векторизованная)
                scalars[0] += weight * ar_Vmac[ii, jj, kk]
                scalars[1] += weight * ar_Vmic[ii, jj, kk]
                scalars[2] += weight * ar_den[ii, jj, kk]
                
                # Векторная интерполяция
                for di in range(dim):
                    vectors[0, di] += weight * ar_d_Vmac[di, ii, jj, kk]
                    vectors[1, di] += weight * ar_d_Vmic[di, ii, jj, kk]
                    vectors[2, di] += weight * ar_d_den[di, ii, jj, kk]

                # Тензорная интерполяция
                for di in range(dim):
                    for dj in range(dim):
                        tensors[0, di, dj] += weight * ar_invM[ii, jj, kk, di, dj]
                        tensors[1, di, dj] += weight * ar_G[ii, jj, kk, di, dj]
                        tensors[2, di, dj] += weight * ar_sqrtG[ii, jj, kk, di, dj]
                        
                        # Тензор 3-го порядка
                        for dk in range(dim):
                            tensor_3d[di, dj, dk] += weight * ar_d_invM[dk, ii, jj, kk, di, dj]

    return (scalars[0], scalars[1], scalars[2],  # Vmac, Vmic, den
            vectors[0], vectors[1], vectors[2],   # dVmac, dVmic, d_den  
            tensors[0], tensors[1], tensors[2],   # invM, G, sqrtG
            tensor_3d)                            # d_invM


###############################################################################
###################### ОБРАТНО СОВМЕСТИМЫЕ ОБЕРТКИ ###########################
###############################################################################

# Алиасы для обратной совместимости
gh_ap3d_jit = gh_interpolate_scalar
gh_ap3d_tens_jit = gh_interpolate_tensor

@nb.njit(fastmath=True, nogil=True, cache=True) 
def gh_ap3d_set_jit(q, qlim, dq, N_q, ar_invM, ar_G, ar_sqrtG, ar_Vmac,
                    ar_Vmic, ar_den, ar_d_invM, ar_d_Vmac, ar_d_Vmic, ar_d_den):
    """
    Обертка для старого интерфейса с МАКСИМАЛЬНОЙ производительностью
    """
    scalar_arrays = (ar_Vmac, ar_Vmic, ar_den)
    vector_arrays = (ar_d_Vmac, ar_d_Vmic, ar_d_den)
    tensor_arrays = (ar_invM, ar_G, ar_sqrtG, ar_d_invM)
    
    results = gh_interpolate_multi_optimized(q, qlim, dq, N_q,
                                           scalar_arrays, vector_arrays, tensor_arrays)
    
    # Распаковка в старый формат
    el_Vmac, el_Vmic, el_den = results[0], results[1], results[2]
    v_dVmac, v_dVmic, d_el_den = results[3], results[4], results[5]  
    t_invM, t_G, t_rootG = results[6], results[7], results[8]
    t_dinvM = results[9]
    
    return t_invM, t_G, t_rootG, el_Vmac, el_Vmic, el_den, d_el_den, t_dinvM, v_dVmac, v_dVmic


# Стандартные обертки без JIT
def gh_ap3d(q, qlim, dq, N_q, matrix):
    return gh_interpolate_scalar(q, qlim, dq, N_q, matrix)

def gh_ap3d_tens(q, qlim, dq, N_q, tensor):
    return gh_interpolate_tensor(q, qlim, dq, N_q, tensor)

def gh_ap3d_set(q, qlim, dq, N_q, *args):
    return gh_ap3d_set_jit(q, qlim, dq, N_q, *args)


###############################################################################
###################### ПРОИЗВОДИТЕЛЬНОСТЬ ####################################
###############################################################################

def benchmark_performance():
    """Сравнение производительности различных подходов"""
    import time
    
    # Тестовые данные
    q = np.array([1.0, 0.1, -0.1])
    qlim = np.array([[0., -0.5, -0.5], [3., 0.5, 0.5]])
    dq = np.array([0.1, 0.1, 0.1])
    N_q = np.array([30, 10, 10])
    
    # Создаем тестовые массивы
    scalar_shape = (30, 10, 10)
    vector_shape = (3, 30, 10, 10)
    tensor_shape = (30, 10, 10, 3, 3)
    
    ar_Vmac = np.random.random(scalar_shape)
    ar_Vmic = np.random.random(scalar_shape) 
    ar_den = np.random.random(scalar_shape)
    ar_d_Vmac = np.random.random(vector_shape)
    ar_d_Vmic = np.random.random(vector_shape)
    ar_d_den = np.random.random(vector_shape)
    ar_invM = np.random.random(tensor_shape)
    ar_G = np.random.random(tensor_shape)
    ar_sqrtG = np.random.random(tensor_shape)
    ar_d_invM = np.random.random((3, 30, 10, 10, 3, 3))
    
    N_tests = 1000
    
    print("🔥 ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ИНТЕРПОЛЯЦИИ")
    print("=" * 50)
    
    # Тест 1: Множественные вызовы (плохой подход)
    start = time.time()
    for _ in range(N_tests):
        # Моя плохая версия - множественные вызовы
        el_Vmac = gh_interpolate_scalar(q, qlim, dq, N_q, ar_Vmac)
        el_Vmic = gh_interpolate_scalar(q, qlim, dq, N_q, ar_Vmic)
        el_den = gh_interpolate_scalar(q, qlim, dq, N_q, ar_den)
    time_multiple = time.time() - start
    
    # Тест 2: Оптимизированная версия (хороший подход)
    start = time.time()
    for _ in range(N_tests):
        results = gh_ap3d_set_jit(q, qlim, dq, N_q, ar_invM, ar_G, ar_sqrtG, 
                                 ar_Vmac, ar_Vmic, ar_den, ar_d_invM, 
                                 ar_d_Vmac, ar_d_Vmic, ar_d_den)
    time_optimized = time.time() - start
    
    speedup = time_multiple / time_optimized
    
    print(f"❌ Множественные вызовы: {time_multiple:.4f}s ({N_tests} итераций)")
    print(f"✅ Оптимизированная версия: {time_optimized:.4f}s ({N_tests} итераций)")
    print(f"🚀 УСКОРЕНИЕ: {speedup:.1f}x")
    print(f"💾 Экономия времени: {(1-1/speedup)*100:.1f}%")


###############################################################################
###################### ЭКСПОРТ #############################################
###############################################################################

__all__ = [
    # Новые оптимизированные функции
    'gh_interpolate_scalar', 'gh_interpolate_tensor', 'gh_interpolate_multi_optimized',
    # JIT-версии
    'gh_ap3d_jit', 'gh_ap3d_tens_jit', 'gh_ap3d_set_jit',
    # Обратно совместимые обертки
    'gh_ap3d', 'gh_ap3d_tens', 'gh_ap3d_set',
    # Утилиты
    'benchmark_performance'
]

if __name__ == "__main__":
    print("✅ Оптимизированная архитектура Gauss-Hermit загружена")
    print("🔧 Единая система интерполяции")
    print("⚡ Максимальная производительность")
    benchmark_performance()

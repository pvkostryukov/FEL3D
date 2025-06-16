"""
Модуль обработки данных и файлового ввода-вывода для FEL3D
Содержит все функции для чтения входных файлов и обработки данных

Извлечено из main.py для улучшения структуры кода
"""

import os
import sys
import platform
import numpy as np
import pandas as pd
import re

from config import (
    DIM as dim, 
    DATA_FOLDERS, 
    DATA_FILES,
    get_element_symbol
)

# Глобальные переменные для совместимости
OS_flag = platform.system() == 'Windows'

###############################################################################
##################### ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ################################
###############################################################################

def odd_axes_elements_adding(matrix, dimensions):
    """
    Adds array elements along axis related with odd q_i
    
    Parameters:
    -----------
    matrix : ndarray
        Input matrix to extend
    dimensions : array_like
        Dimensions of the matrix
        
    Returns:
    --------
    ndarray
        Extended matrix with odd elements added
    """
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
    """
    Additional utility to reading prompts from Fourier data files
    
    Parameters:
    -----------
    line : ndarray
        Data line from file
    dim : int
        Dimension of the tensor
        
    Returns:
    --------
    ndarray
        Reconstructed tensor matrix
    """
    mat = np.zeros((4, 4))
    promt = line.copy()
    for i in range(4):
        mat[i, i:] = promt[:4-i]
        mat[i+1:, i] = promt[1:4-i]
        promt = promt[4-i:]
    return mat[-dim:, -dim:]


###############################################################################
##################### ФУНКЦИИ ЧТЕНИЯ ФАЙЛОВ ##################################
###############################################################################

def fourier_file_data(data_file: str, path: str = None):
    """
    Extract input data calculated within framework of Fourier nuclear shape
    parametrization consisted all parameters, i.e transport coefficients
    
    Parameters:
    -----------
    data_file : str
        Name of the Fourier data file
    path : str, optional
        Base path for data files
        
    Returns:
    --------
    tuple
        (N_q, dq, qlim, m_0, f_0, bs, bc, bk, bf, r12, bx, vol, c, rn)
    """
    if path is None:
        path = os.getcwd()
        
    exact_place = os.getcwd()  # Save current directory
    
    # Navigate to Fourier data folder
    if OS_flag:
        os.chdir(path + '\\' + DATA_FOLDERS['fourier'] + '\\')
    else:
        os.chdir(path + '/' + DATA_FOLDERS['fourier'] + '/')

    try:
        # Read data from file
        data = [line for line in open(data_file, 'r').readlines()
                if line[0] not in ('#', '\n') or line[0].isalpha()]
    except FileNotFoundError:
        os.chdir(exact_place)
        raise FileNotFoundError(f"Fourier data file {data_file} not found")
    finally:
        os.chdir(exact_place)  # Always return to original directory

    # Parse header information
    qlim_line = np.array([float(el) for el in data[3].split()[1:]])
    dq_line = np.array([float(el) for el in data[4].split()[1:]])
    nq_line = np.array([int(el) for el in data[5].split()[1:]])

    which_q = np.where(nq_line - 1 > 0)[0]
    dim_data = len(which_q)
    N_q, dq, qlim = nq_line[which_q], dq_line[which_q], qlim_line[which_q]
    qlim = np.stack([qlim, qlim + (N_q - 1) * dq])

    # Initialize data arrays
    bs = np.empty(N_q)
    bc = np.empty(N_q)
    bk = np.empty(N_q)
    bx = np.empty(N_q)
    bf = np.empty(N_q)
    r12 = np.empty(N_q)
    rn = np.empty(N_q)
    vol = np.empty(N_q)
    c = np.empty(N_q)
    m_0 = np.empty(np.append(N_q, (dim_data, dim_data)))
    f_0 = np.empty(np.append(N_q, (dim_data, dim_data)))

    # Parse data
    data = data[12:]

    for i, el in enumerate(data[::3]):
        dat_line = np.array(list(map(float, el.split())))
        ind = tuple(np.round((dat_line[which_q] - qlim[0]) / dq).astype(int))
        bs[ind], bc[ind], bk[ind] = dat_line[6:9]
        bf[ind], r12[ind], bx[ind] = dat_line[10:13]
        vol[ind], c[ind], rn[ind] = dat_line[15:]
        
        dat_line = np.array(list(map(float, data[3 * i + 1].split()[:-1])))
        m_0[ind] = tensor_data_reader(dat_line, dim_data)
        
        dat_line = np.array(list(map(float, data[3 * i + 2].split()[:-1])))
        f_0[ind] = tensor_data_reader(dat_line, dim_data)

    # Add odd elements
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


def potential_reader(A, Z, N_q, dq, qlim, file_extension='.1', path=None):
    """
    Extract input data calculated within framework of Fourier nuclear shape
    parametrization consisted all parameters, i.e transport coefficients
    
    Parameters:
    -----------
    A : float
        Mass number
    Z : float  
        Atomic number
    N_q : array_like
        Grid dimensions
    dq : array_like
        Grid spacing
    qlim : array_like
        Grid limits
    file_extension : str
        File extension for potential files
    path : str, optional
        Base path for data files
        
    Returns:
    --------
    tuple
        (V_macro, V_micro) - macroscopic and microscopic potentials
    """
    if path is None:
        path = os.getcwd()
        
    isotope_name = get_element_symbol(Z) + '-' + str(int(A))
    isotope_file = isotope_name + file_extension
    
    exact_place = os.getcwd()
    
    # Navigate to PES data folder
    if OS_flag:
        os.chdir(path + '\\' + DATA_FOLDERS['pes'] + '\\')
    else:
        os.chdir(path + '/' + DATA_FOLDERS['pes'] + '/')

    try:
        if isotope_file not in os.listdir():
            raise FileNotFoundError(f'No file {isotope_file} in PES data directory')
        
        data = open(isotope_file, 'r').readlines()
    finally:
        os.chdir(exact_place)

    # Parse file header
    q_dim_data = [i for i in data[2].split() if i[0] in ['q', 'Q']]
    is_4d = len(q_dim_data) == len(N_q)  # check 4D or 3D case

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

    # Parse potential data
    for i, el in enumerate(data[:n_q_file.prod()]):
        dat_line = np.array(list(map(float, el.split())))
        idx = tuple(np.round((dat_line[q_idx] - qlim_file) / dq).astype(int))
        V_macro[idx] = dat_line[eld_idx]
        V_micro[idx] = dat_line[e_tot_idx] - V_macro[idx]

    V_macro = odd_axes_elements_adding(V_macro, q_idx)
    V_micro = odd_axes_elements_adding(V_micro, q_idx)

    return V_macro, V_micro


def st_pnt_def(a_nuc, z_nuc, saddle_type: str = '2sad', path=None):
    """
    Extraction of saddle point from database sad_pnt_crds.xlsx
    
    Parameters:
    -----------
    a_nuc : float
        Mass number
    z_nuc : float
        Atomic number  
    saddle_type : str
        Type of saddle point ('2sad' or ' 2min')
    path : str, optional
        Base path for data files
        
    Returns:
    --------
    ndarray
        Coordinates of the saddle point
    """
    if path is None:
        path = os.getcwd()
        
    saddle_file = DATA_FILES['saddle_points']
    
    if saddle_file not in os.listdir(path):
        print('There no library file on main folder.' +
              ' The program will be aborted')
        return sys.exit()
    
    data = pd.read_excel(os.path.join(path, saddle_file), 
                         sheet_name='Z'+str(int(z_nuc)),
                         engine='openpyxl')
    
    # Get dimension from global config
    list_of_q = ['q' + str(i) for i in range(5 - dim, 5)]
    type_of_point = '_2_min' if saddle_type == ' 2min' else '_2_sad'
    list_of_q = [i + type_of_point for i in list_of_q]
    list_of_q.insert(0, 'A')
    
    data = data.loc[:, list_of_q].to_numpy()
    crd = data[np.where(a_nuc == data[:, 0]), 1:].flatten()
    
    if np.size(crd) == 0:
        print('There no information about this isotope.' +
              ' The program will be aborted')
        return sys.exit()
    else:
        return crd


def exp_res_aut(file_name, dir_path):
    """
    Read experimental results automatically from file
    
    Parameters:
    -----------
    file_name : str
        Name of the experimental data file
    dir_path : str
        Directory path containing the file
        
    Returns:
    --------
    tuple
        (nucl_bins, nucl_yield) - experimental mass distribution
    """
    exact_place = os.getcwd()
    
    # Navigate to experimental data folder
    if OS_flag:
        os.chdir(dir_path + '\\' + DATA_FOLDERS['experimental'] + '\\')
    else:
        os.chdir(dir_path + '/' + DATA_FOLDERS['experimental'] + '/')

    try:
        # Extract mass number from filename
        A = ''.join([num for num in file_name[:5] if num.isdigit()])
        A = int(A)
        
        with open(file_name, 'r') as file:
            text = file.readlines()
    finally:
        os.chdir(exact_place)

    # Parse experimental data
    data = [line for line in text if line[0] not in ('\n', '#')
            and not line[0].isalpha()]
    
    nucl_yield = []
    nucl_bins = []
    
    for line in data:
        line_list = list(map(float, line.split()))
        if 0 <= line_list[0] <= A:
            nucl_bins.append(line_list[0])
            nucl_yield.append(line_list[1])
    
    # Normalize yields if needed
    if sum(nucl_yield) > 3:
        nucl_yield = [i / 100 for i in nucl_yield]
    
    # Handle asymmetric distributions
    if nucl_bins[0] >= int(A // 2) or nucl_bins[-1] <= int(A // 2) + 1:
        nucl_bins1 = [A - i for i in reversed(nucl_bins[:-1])]
        nucl_yield1 = nucl_yield[1:].copy()
        nucl_yield1.reverse()
        nucl_bins = nucl_bins1 + nucl_bins
        nucl_yield = nucl_yield1 + nucl_yield
    
    return np.array(nucl_bins), np.array(nucl_yield)


###############################################################################
##################### ФУНКЦИИ ОБРАБОТКИ ДАННЫХ ###############################
###############################################################################

def validate_input_file(file_path):
    """
    Validate input Excel file format (soft validation)

    Parameters:
    -----------
    file_path : str
        Path to input file

    Returns:
    --------
    bool
        True if file can be read
    """
    try:
        data = pd.read_excel(file_path)
        if len(data) == 0:
            print("Warning: Input file is empty")
            return False
        return True
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False


def load_input_data(file_path=None):
    """
    Load input data from Excel file (backward compatible)

    Parameters:
    -----------
    file_path : str, optional
        Path to input file (defaults to 'input.xlsx')

    Returns:
    --------
    DataFrame
        Loaded input data
    """
    if file_path is None:
        file_path = DATA_FILES['input']

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Error: there no {file_path} file!')

    try:
        # Простая загрузка без строгой валидации для обратной совместимости
        return pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f'Cannot read input file {file_path}: {e}')


def prepare_data_paths(base_path=None):
    """
    Prepare and validate data directory paths

    Parameters:
    -----------
    base_path : str, optional
        Base directory path

    Returns:
    --------
    dict
        Dictionary of validated data paths
    """
    if base_path is None:
        base_path = os.getcwd()

    paths = {}
    for key, folder in DATA_FOLDERS.items():
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            print(f"Warning: Data folder {folder} not found at {path}")
        paths[key] = path

    return paths


###############################################################################
##################### ЭКСПОРТ РЕЗУЛЬТАТОВ ####################################
###############################################################################

def save_results_to_excel(output_data, file_name, result_path=None):
    """
    Save calculation results to Excel file

    Parameters:
    -----------
    output_data : dict
        Dictionary containing output data sheets
    file_name : str
        Output file name
    result_path : str, optional
        Path to results directory
    """
    if result_path is None:
        result_path = os.path.join(os.getcwd(), DATA_FOLDERS['results'])

    # Create results directory if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    full_path = os.path.join(result_path, file_name)

    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
        for sheet_name, data in output_data.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Handle other data types
                pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)


###############################################################################
##################### ТЕСТИРОВАНИЕ МОДУЛЯ ####################################
###############################################################################

if __name__ == "__main__":
    print("Data handling module loaded successfully")

    # Test data paths
    paths = prepare_data_paths()
    print("Available data paths:")
    for key, path in paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {key}: {path} {exists}")
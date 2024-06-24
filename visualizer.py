###############################################################################
##################### CALLING USED LIBRARIES & PACKAGES #######################
###############################################################################

import os
import re
import sys
import platform

import pandas as pd
import numpy  as np

from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.ticker as ticker

###############################################################################

user_path = os.getcwd()
OS_flag = platform.system() == 'Windows'

yes = ('yes', 'y', 'д', 'да', 'tak', 'tak!')
tab_clrs = list(mcolors.TABLEAU_COLORS.values())

###############################################################################

###############################################################################
############################# FUNCTION SECTION ################################
###############################################################################

def folder_reader():
    result_path = user_path
    result_path += '\\Result\\' if OS_flag else '/Result/'
    if not os.path.isdir(result_path):
        raise RuntimeError('There is no folder Result in the directory root!')
    os.chdir(result_path)

    inpts = []
    inps_paths = []

    grab_in_folder = True

    while grab_in_folder: 
        data_dir = input('Please write a folder in format d-m-y: ')
        fdir = result_path
        fdir += f'\\{data_dir}\\' if OS_flag else f'/{data_dir}/'
        if not os.path.isdir(fdir):
            print('There are no such folder!')
            continue
        os.chdir(fdir)

        folder_data = []
        i = 1

        for el in sorted(os.listdir()):
            if '.xlsx' in el:
                print(f'{i}. ' + el)
                i += 1
                folder_data.append(el)
        if folder_data == []:
            print('No xlsx files here. Please, choose another directory.')
            continue

        print('\nChoose files: ')

        while True:
            inpt = input()
            if inpt == 'exit':
                sys.exit()
            elif inpt in ('', '.', ' ', '\n', 'all', 'cd', 'chdir'):
                if inpt in ('', '.', ' ', '\n'):
                    grab_in_folder = False
                elif inpt == 'all':
                    inpts += folder_data
                    inps_paths += [fdir for _ in folder_data]
                    grab_in_folder = not (input("That's all?(y/n) ") in yes)
                break
            elif inpt.isdigit():
                number = int(inpt)
                if number > len(folder_data):
                    print('Invalid number. There no such position in list.' +\
                          ' Try again!')
                    continue
                inpts.append(folder_data[number - 1])
                inps_paths.append(fdir)
            elif re.sub('[-]', '', inpt).isdigit():
                set_of_f = np.array(re.sub('[-]', ' ', inpt).split(),
                                    dtype=int)
                
                if any(set_of_f > len(folder_data)):
                    print('Unable to select files in the given interval.' + 
                          ' Try again!')
                    continue

                set_of_f[set_of_f == set_of_f.min()] -= 1
                normal_order = set_of_f[0] <= set_of_f[1]
                if normal_order:
                    list_of_num = np.arange(*set_of_f, dtype=int)
                else:
                    set_of_f -= 1
                    list_of_num = np.arange(*set_of_f, dtype=int, step=-1)
                for num_file in list_of_num:
                    inpts.append(folder_data[num_file])
                    inps_paths.append(fdir)

            elif re.sub('[, ]', '', inpt).isdigit():
                list_of_files = np.array(re.sub('[, ]', ' ', inpt).split(),
                                         dtype=int)
                list_of_files = list_of_files[list_of_files <= len(folder_data)]
                for el in list_of_files:
                    inpts.append(folder_data[int(el) - 1])
                    inps_paths.append(fdir)
            else:
                if inpt in folder_data[num_file]:
                    inpts.append(inpt)
                    inps_paths.append(fdir)

    if inpts == []:
        return inpts

    data_list = []
    global_info = []
    main_sheets = []
    second_sheets = []
    third_sheets = []

    for f_n, drc in zip(inpts, inps_paths):
        book_dat = []
        xl = pd.ExcelFile(drc + f_n)
        for sheet in xl.sheet_names:
            book_dat.append(pd.read_excel(drc + f_n, sheet, engine='openpyxl'))
        data_list.append(book_dat)
        main_sheets.append(book_dat[0])
        more_1p = len(book_dat) > 1
        second_sheets.append(book_dat[1] if more_1p else [])
        third_sheets.append(book_dat[2] if more_1p else [])

    fmy_init, fcy_init = [], []
    fmy_prime, fcy_prime = [], []

    for page in third_sheets:
        if type(page) != list:
            fmy_init.append(page[["Af","Y(Af)"]].dropna().to_numpy())
            fcy_init.append(page[["Zf","Y(Zf)"]].dropna().to_numpy())
            fmy_prime.append(page[["A'f","Y(A'f)"]].dropna().to_numpy())
            fcy_prime.append(page[["Z'f","Y(Z'f)"]].dropna().to_numpy())
        else:
            fmy_init.append(page)
            fcy_init.append(page)
    return fmy_init, fmy_prime, fcy_init, fcy_prime


def axis_style(ax, major_length=10, major_width=3):

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)

    ax.tick_params(axis='both',
                   pad=10,
                   which='major',
                   direction='in',
                   labelsize=axes_font,
                   length=major_length,
                   width=major_width,
                   bottom=True,
                   top=True,
                   left=True,
                   right=True,
                   labelbottom=True,
                   labelleft=True,
                   zorder=0)

    ax.tick_params(axis='both',
                   which='minor',
                   direction='in',
                   length=major_length / 1.5,
                   width=major_width / 2,
                   bottom=True,
                   top=True,
                   left=True,
                   right=True,
                   zorder=0)

    # return ax


def axes_style(ax, major_length=10, major_width=3):

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)

    ax.tick_params(axis='both',
                   pad=10,
                   which='major',
                   direction='in',
                   labelsize=axes_font,
                   length=major_length,
                   width=major_width,
                   bottom=True,
                   top=True,
                   left=True,
                   right=True,
                   zorder=0)

    ax.tick_params(axis='both',
                   which='minor',
                   direction='in',
                   length=major_length / 1.5,
                   width=major_width / 2,
                   bottom=True,
                   top=True,
                   left=True,
                   right=True,
                   zorder=0)

    # return ax


def set_ax_ticks_rule(ax, x_ticks):
    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylim(1e-5)
    ax.set_xlim(*x_range)


if __name__ == "__main__":

    fmds, fmds_aft, fcds, fcds_aft = folder_reader()
    kind_of_plot = input("Do you want multiple plots?(y/n) ") in yes

    plt.rc('text', usetex=True)
    N_tot = len(fmds)
    if kind_of_plot:
        n_column = N_tot if N_tot <= 6 else 6
        n_row = 1 if N_tot <= 6 else N_tot // 6 + 1
    else:
        n_row, n_column = 1, 1
    font_scale_factor = 1 + .2 * (n_column - 1)
    axes_font = 30 * font_scale_factor # 15 * sqr5
    sup_font = 1.5 * axes_font
    line_width = 3 * font_scale_factor
    major_tick_length, major_tick_width = 4 * line_width, 1.5 * line_width

    for dist in ((fmds, fmds_aft), (fcds, fcds_aft)):

        fig, axs = plt.subplots(n_row, n_column, dpi=200,
                                figsize=(7 * n_column, 7.5 * n_row),
                                sharex=True, sharey=True,
                                gridspec_kw={'hspace':0, 'wspace':0,}
                                )

        x_suplabel, y_suplabel = (r'$\rm A_f$', r'$\rm Y(A_f),~\%$')\
            if dist[0][0] is fmds[0] else (r'$\rm Z_f$', r'$\rm Y(Z_f),~\%$')

        if not type(axs) is np.ndarray:
            axs.set_xlabel(x_suplabel, fontsize=sup_font)
            axs = np.array([axs,])
            axes_style_fun = axis_style
        else:
            axes_style_fun = axes_style
            fig.supxlabel(x_suplabel, size=sup_font, in_layout=True,
                          ha='center')

        if n_row == 1:
            axs[0].set_ylabel(y_suplabel, fontsize=sup_font)
        else:
            fig.supylabel(y_suplabel, size=sup_font, in_layout=True,
                          ha='right', va='center')

        xlims = np.array([[np.take(d[:, 0][d[:, 1] > 5e-4], [0, -1])
                          for d in dstr] for dstr in dist])
        x_range = (xlims.min(), xlims.max())

        for i, ax in enumerate(axs.flatten()):
            if i >= N_tot:
                ax.remove()
                continue
            if not kind_of_plot:
                for j in range(N_tot):
                    ax.plot(dist[0][j][:, 0], 100 * dist[0][j][:, 1],
                            lw=line_width, c=tab_clrs[j], alpha= .4 + .05 * j)
                    ax.plot(dist[1][j][:, 0], 100 * dist[1][j][:, 1],
                            lw=line_width, c=tab_clrs[j], alpha= .4 + .05 * j)
            else:
                ax.plot(dist[0][i][:, 0], 100 * dist[0][i][:, 1], lw=line_width)
                ax.plot(dist[1][i][:, 0], 100 * dist[1][i][:, 1], lw=line_width)

            axes_style_fun(ax, major_tick_length, major_tick_width)
            set_ax_ticks_rule(ax, x_range)

        fig.tight_layout()
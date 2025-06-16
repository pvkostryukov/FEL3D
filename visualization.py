"""
Модуль визуализации результатов для FEL3D
Академический стиль, фокус на массовом распределении

Может использоваться как отдельный скрипт или импортироваться как модуль
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import argparse
from typing import Dict, List, Optional
from scipy.ndimage import gaussian_filter1d


###############################################################################
##################### НАСТРОЙКИ АКАДЕМИЧЕСКОГО СТИЛЯ ########################
###############################################################################

# Строгий академический стиль для публикаций
ACADEMIC_STYLE = {
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Шрифты
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    
    # Сетка и оси
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
    
    # Тики
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 6,
    'ytick.minor.size': 6,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    
    # Линии и маркеры
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'patch.linewidth': 1.0,
    
    # Цвета (консервативная палитра)
    'axes.prop_cycle': plt.cycler('color', [
        '#000000',  # черный
        '#FF0000',  # красный  
        '#0000FF',  # синий
        '#008000',  # зеленый
        '#FF8C00',  # оранжевый
        '#8B008B',  # фиолетовый
        '#8B4513',  # коричневый
        '#2F4F4F'   # темно-серый
    ]),
    
    # LaTeX
    'text.usetex': False,  # Отключаем LaTeX по умолчанию для совместимости
    'mathtext.fontset': 'stix',
    
    # Легенда
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    
    # Оставляем все границы
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
}

# Применяем стиль
plt.rcParams.update(ACADEMIC_STYLE)

###############################################################################
##################### КЛАСС АНАЛИЗАТОРА РЕЗУЛЬТАТОВ ########################
###############################################################################

class FEL3DAnalyzer:
    """
    Основной класс для анализа и визуализации результатов FEL3D
    """
    
    def __init__(self, excel_file: str):
        """
        Инициализация анализатора
        
        Parameters:
        -----------
        excel_file : str
            Путь к Excel файлу с результатами
        """
        self.excel_file = excel_file
        self.data = {}
        self.metadata = {}
        self.load_data()
        
    def load_data(self):
        """Загрузка данных из Excel файла"""
        if not os.path.exists(self.excel_file):
            raise FileNotFoundError(f"File {self.excel_file} not found")
            
        try:
            # Загрузка всех листов
            excel_data = pd.read_excel(self.excel_file, sheet_name=None, engine='openpyxl')
            
            self.data['trajectories'] = excel_data.get('Sheet1', pd.DataFrame())
            self.data['parameters'] = excel_data.get('Sheet2', pd.DataFrame())
            self.data['yields'] = excel_data.get('Sheet3', pd.DataFrame())
            
            # Извлечение метаданных из имени файла
            self.extract_metadata()
            
            print(f"✓ Data loaded: {len(self.data['trajectories'])} trajectories")
            
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
    
    def extract_metadata(self):
        """Извлечение метаданных из имени файла"""
        filename = Path(self.excel_file).stem
        
        # Парсинг имени файла (например: u235_e(10.5)_n3000_dt001_...)
        parts = filename.split('_')
        
        # Извлечение изотопа (первая часть)
        if parts:
            isotope_part = parts[0].lower()
            # Парсинг изотопа (например: u235, pu239, cf252)
            element = ''.join([c for c in isotope_part if c.isalpha()]).upper()
            mass = ''.join([c for c in isotope_part if c.isdigit()])
            if element and mass:
                self.metadata['isotope'] = f"{element}-{mass}"
                self.metadata['mass_number'] = int(mass)
            else:
                self.metadata['isotope'] = isotope_part.upper()
        
        # Остальные параметры
        for part in parts:
            if part.startswith('e(') and part.endswith(')'):
                self.metadata['energy'] = float(part[2:-1])
            elif part.startswith('n') and part[1:].isdigit():
                self.metadata['n_trajectories'] = int(part[1:])
            elif part.startswith('dt'):
                self.metadata['dt'] = float('0.' + part[2:])

    def get_mass_distribution_data(self) -> Dict:
        """
        Получение данных для массового распределения из готовых столбцов
        
        Returns:
        --------
        dict
            Словарь с массами и выходами (до и после эмиссии нейтронов)
        """
        if 'yields' not in self.data or self.data['yields'].empty:
            return {}
            
        yields_data = self.data['yields']
        result = {}
        
        # Ищем данные до эмиссии нейтронов
        if 'Af' in yields_data.columns and 'Y(Af)' in yields_data.columns:
            # Убираем NaN значения
            mask = ~(yields_data['Af'].isna() | yields_data['Y(Af)'].isna())
            result['primary'] = {
                'masses': yields_data['Af'][mask].values,
                'yields': yields_data['Y(Af)'][mask].values,
                'label': 'Primary fragments'
            }
        
        # Ищем данные после эмиссии нейтронов
        if "A'f" in yields_data.columns and "Y(A'f)" in yields_data.columns:
            mask = ~(yields_data["A'f"].isna() | yields_data["Y(A'f)"].isna())
            result['secondary'] = {
                'masses': yields_data["A'f"][mask].values,
                'yields': yields_data["Y(A'f)"][mask].values,
                'label': 'After neutron emission'
            }
        
        return result

###############################################################################
##################### КЛАСС ПОСТРОЕНИЯ ГРАФИКОВ ############################
###############################################################################

class FEL3DPlotter:
    """Класс для создания академических графиков"""
    
    def __init__(self, analyzer: FEL3DAnalyzer):
        self.analyzer = analyzer
        self.data = analyzer.data
        self.metadata = analyzer.metadata

    def plot_mass_distribution(self, save_path: Optional[str] = None,
                             show_statistics: bool = True) -> None:
        """
        График массового распределения фрагментов деления

        Parameters:
        -----------
        save_path : str, optional
            Путь для сохранения графика
        show_statistics : bool
            Показывать ли статистические данные на графике
        """
        mass_data = self.analyzer.get_mass_distribution_data()

        if not mass_data:
            print("No mass distribution data available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Цвета и стили для разных типов данных
        colors = ['black', 'red']
        markers = ['o', 's']
        linestyles = ['-', '--']

        plot_data = []

        # Построение графиков для каждого типа данных
        for i, (data_type, data_info) in enumerate(mass_data.items()):
            masses = data_info['masses']
            yields = data_info['yields']
            label = data_info['label']

            # Сортируем данные по массам
            sort_idx = np.argsort(masses)
            masses_sorted = masses[sort_idx]
            yields_sorted = yields[sort_idx]

            # Основной график
            ax.plot(masses_sorted, yields_sorted,
                   color=colors[i], marker=markers[i], linestyle=linestyles[i],
                   linewidth=2, markersize=4, markerfacecolor='white',
                   markeredgewidth=1.5, label=label)

            # Заливка области под кривой (только для первичных фрагментов)
            if data_type == 'primary':
                ax.fill_between(masses_sorted, yields_sorted, alpha=0.2, color='lightgray')

            plot_data.append((masses_sorted, yields_sorted, label))

        # Настройка осей с LaTeX подписями
        ax.set_xlabel(r'Fragment mass number $\rm A_f$', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')
        ax.set_ylabel(r'Fission yield $\rm Y(A_f)$ (%)', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')

        # Настройка тиков и minorticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Составное ядро в углу (вместо заголовка)
        isotope = self.metadata.get('isotope', 'Unknown')
        energy = self.metadata.get('energy', 'N/A')

        # Парсим изотоп для LaTeX формата
        if '-' in isotope:
            element, mass_num = isotope.split('-')
            nucleus_text = rf'$\rm ^{{{mass_num}}}{element}$'
        else:
            nucleus_text = isotope

        if energy != 'N/A':
            nucleus_text += rf', $E^*={energy}$, $\text{{MeV}}$'

        ax.text(0.98, 0.98, nucleus_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=ACADEMIC_STYLE['figure.titlesize'],
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="black", alpha=0.9))

        # Статистика на графике
        if show_statistics and plot_data:
            # Используем данные первичных фрагментов для статистики
            primary_data = plot_data[0]
            masses_for_stats = primary_data[0]
            yields_for_stats = primary_data[1]

            # Находим пики с улучшенным алгоритмом
            peak_indices = self._find_peaks(yields_for_stats, masses_for_stats)

            if peak_indices:
                # Определяем общую массу ядра для классификации
                total_mass = self.metadata.get('mass_number', None)
                fission_mode = self._classify_fission_mode(peak_indices, masses_for_stats, total_mass)

                # Отмечаем все найденные пики одним цветом
                for peak_idx in peak_indices:
                    peak_mass = masses_for_stats[peak_idx]
                    ax.axvline(peak_mass, color='blue', linestyle=':', alpha=0.7, linewidth=2)

                # Формируем текст со статистикой
                n_traj = self.metadata.get('n_trajectories', 'N/A')
                stats_text = f'Fission mode: {fission_mode}\n'

                if len(peak_indices) == 1:
                    peak_mass = masses_for_stats[peak_indices[0]]
                    stats_text += rf'Peak: $A = {peak_mass:.0f}$' + '\n'
                elif len(peak_indices) == 2:
                    light_mass = masses_for_stats[peak_indices[0]]
                    heavy_mass = masses_for_stats[peak_indices[1]]
                    stats_text += rf'Light peak: $A_L = {light_mass:.0f}$' + '\n'
                    stats_text += rf'Heavy peak: $A_H = {heavy_mass:.0f}$' + '\n'
                else:  # 3 пика
                    for i, peak_idx in enumerate(peak_indices):
                        peak_mass = masses_for_stats[peak_idx]
                        if i == 0:
                            stats_text += rf'Light peak: $A_L = {peak_mass:.0f}$' + '\n'
                        elif i == 1:
                            stats_text += rf'Symmetric: $A_S = {peak_mass:.0f}$' + '\n'
                        else:
                            stats_text += rf'Heavy peak: $A_H = {peak_mass:.0f}$' + '\n'

                stats_text += f'Trajectories: $N = {n_traj}$'

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="black", alpha=0.9))
        # Легенда (если есть несколько типов данных)
        if len(mass_data) > 1:
            ax.legend(loc='upper center', frameon=True, fancybox=False,
                     edgecolor='black', facecolor='white')

        # Настройка сетки и внешнего вида
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Определяем диапазон осей
        all_masses = np.concatenate([data['masses'] for data in mass_data.values()])
        all_yields = np.concatenate([data['yields'] for data in mass_data.values()])

        ax.set_xlim(all_masses.min() - 2, all_masses.max() + 2)
        ax.set_ylim(0, all_yields.max() * 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_mass_distribution.png", dpi=300, bbox_inches='tight')
            print(f"✓ Mass distribution saved: {save_path}_mass_distribution.png")

        plt.show()

    def _find_peaks(self, data: np.ndarray, masses: np.ndarray = None,
                    min_height_ratio: float = 0.1, min_distance: int = 8) -> List[int]:
        """
        Улучшенный поиск пиков в массовом распределении

        Parameters:
        -----------
        data : np.ndarray
            Данные выходов фрагментов
        masses : np.ndarray, optional
            Соответствующие массовые числа
        min_height_ratio : float
            Минимальная относительная высота пика (от максимума)
        min_distance : int
            Минимальное расстояние между пиками

        Returns:
        --------
        List[int]
            Индексы найденных пиков
        """
        if len(data) < 3:
            return []

        # Сглаживание данных для уменьшения шума
        smoothed_data = gaussian_filter1d(data, sigma=1.0)

        # Поиск локальных максимумов
        peaks_candidates = []
        for i in range(1, len(smoothed_data) - 1):
            if (smoothed_data[i] > smoothed_data[i - 1] and
                    smoothed_data[i] > smoothed_data[i + 1]):
                peaks_candidates.append(i)

        if not peaks_candidates:
            return []

        # Фильтрация по высоте
        max_height = np.max(smoothed_data)
        min_height = max_height * min_height_ratio

        significant_peaks = []
        for peak_idx in peaks_candidates:
            if smoothed_data[peak_idx] >= min_height:
                significant_peaks.append((peak_idx, smoothed_data[peak_idx]))

        if not significant_peaks:
            return []

        # Сортируем по высоте (убывание)
        significant_peaks.sort(key=lambda x: x[1], reverse=True)

        # Применяем минимальное расстояние между пиками
        final_peaks = []
        for peak_idx, height in significant_peaks:
            # Проверяем расстояние до уже выбранных пиков
            too_close = False
            for existing_peak in final_peaks:
                if abs(peak_idx - existing_peak) < min_distance:
                    too_close = True
                    break

            if not too_close:
                final_peaks.append(peak_idx)

                # Ограничиваем максимальным количеством пиков (3)
                if len(final_peaks) >= 3:
                    break

        # Сортируем финальные пики по позиции (слева направо)
        final_peaks.sort()

        return final_peaks

    def _classify_fission_mode(self, peaks: List[int], masses: np.ndarray,
                               total_mass: float = None) -> str:
        """
        Классификация типа деления по количеству и положению пиков

        Parameters:
        -----------
        peaks : List[int]
            Индексы пиков
        masses : np.ndarray
            Массовые числа
        total_mass : float, optional
            Общая масса делящегося ядра

        Returns:
        --------
        str
            Тип деления: 'symmetric', 'asymmetric', 'mixed'
        """
        if len(peaks) == 0:
            return 'unknown'
        elif len(peaks) == 1:
            # Один пик - проверяем, симметричное ли деление
            peak_mass = masses[peaks[0]]
            if total_mass and abs(peak_mass - total_mass / 2) < 10:
                return 'symmetric'
            else:
                return 'asymmetric (single)'
        elif len(peaks) == 2:
            return 'asymmetric'
        else:  # 3 пика
            return 'mixed'

    def plot_deformation_space(self, save_path: Optional[str] = None):
        """График в пространстве деформаций (дополнительная опция)"""
        if self.data['trajectories'].empty:
            print("No trajectory data available")
            return
            
        traj = self.data['trajectories']
        
        if not all(col in traj.columns for col in ['q2', 'q3']):
            print("Missing deformation coordinates")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Основной scatter plot
        scatter = ax.scatter(traj['q2'], traj['q3'], 
                           c='black', alpha=0.6, s=20, edgecolors='none')
        
        ax.set_xlabel(r'$q_2$ (elongation)', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')
        ax.set_ylabel(r'$q_3$ (asymmetry)', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')
        
        # Настройка тиков и minorticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        # Составное ядро в углу
        isotope = self.metadata.get('isotope', 'Unknown')
        if '-' in isotope:
            element, mass_num = isotope.split('-')
            nucleus_text = rf'$^{{{mass_num}}}{element}$ trajectories in deformation space'
        else:
            nucleus_text = f'{isotope} trajectories in deformation space'
        
        ax.text(0.98, 0.98, nucleus_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                        edgecolor="black", alpha=0.9))
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_deformation.png", dpi=300, bbox_inches='tight')
            print(f"✓ Deformation plot saved: {save_path}_deformation.png")
        
        plt.show()

    def plot_temporal_evolution(self, save_path: Optional[str] = None):
        """График временной эволюции (дополнительная опция)"""
        traj = self.data['trajectories']
        
        if traj.empty or 'time' not in traj.columns:
            print("No temporal data available")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Время деления
        ax1.hist(traj['time'], bins=30, color='lightgray', edgecolor='black', alpha=0.7)
        ax1.set_xlabel(r'Fission time $t$ (fm/c)', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')
        ax1.set_ylabel(r'Count $N$', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')
        
        # Настройка тиков
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        # Составное ядро для первого графика
        isotope = self.metadata.get('isotope', 'Unknown')
        if '-' in isotope:
            element, mass_num = isotope.split('-')
            nucleus_text = rf'$^{{{mass_num}}}{element}$ fission time distribution'
        else:
            nucleus_text = f'{isotope} fission time distribution'
        
        ax1.text(0.98, 0.98, nucleus_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                         edgecolor="black", alpha=0.9))
        
        ax1.grid(True, alpha=0.3)
        
        # Температурная эволюция (если есть)
        if 'Temperature' in traj.columns:
            ax2.scatter(traj['time'], traj['Temperature'], 
                       c='black', alpha=0.6, s=15)
            ax2.set_xlabel(r'Time $t$ (fm/c)', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')
            ax2.set_ylabel(r'Temperature $T$ (MeV)', fontsize=ACADEMIC_STYLE['axes.labelsize'], fontweight='bold')
            
            # Настройка тиков
            ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            
            ax2.text(0.98, 0.98, 'Temperature evolution', transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                             edgecolor="black", alpha=0.9))
            
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_temporal.png", dpi=300, bbox_inches='tight')
            print(f"✓ Temporal plot saved: {save_path}_temporal.png")
        
        plt.show()

    def create_summary_report(self, save_path: Optional[str] = None):
        """Краткий сводный отчет"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Массовое распределение (главный график)
        ax1 = fig.add_subplot(gs[0, :])
        mass_data = self.analyzer.get_mass_distribution_data()
        
        if mass_data:
            # Отображаем данные
            colors = ['black', 'red']
            markers = ['o', 's']
            linestyles = ['-', '--']
            
            for i, (data_type, data_info) in enumerate(mass_data.items()):
                masses = data_info['masses']
                yields = data_info['yields']
                label = data_info['label']
                
                # Сортируем данные
                sort_idx = np.argsort(masses)
                masses_sorted = masses[sort_idx]
                yields_sorted = yields[sort_idx]
                
                ax1.plot(masses_sorted, yields_sorted, 
                        color=colors[i], marker=markers[i], linestyle=linestyles[i],
                        linewidth=2, markersize=3, label=label)
                
                # Заливка только для первичных фрагментов
                if data_type == 'primary':
                    ax1.fill_between(masses_sorted, yields_sorted, alpha=0.2, color='lightgray')
            
            ax1.set_xlabel(r'Fragment mass number $\rm A_f$', fontweight='bold', fontsize=ACADEMIC_STYLE['axes.labelsize'])
            ax1.set_ylabel(r'Fission yield $\rm Y(A_f)$ (%)', fontweight='bold', fontsize=ACADEMIC_STYLE['axes.labelsize'])
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.grid(True, alpha=0.3)
            
            if len(mass_data) > 1:
                ax1.legend(loc='upper right', frameon=True)
        
        # Деформационное пространство
        traj = self.data['trajectories']
        if not traj.empty and 'q2' in traj.columns and 'q3' in traj.columns:
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.scatter(traj['q2'], traj['q3'], c='black', alpha=0.6, s=15)
            ax2.set_xlabel(r'$q_2$', fontweight='bold', fontsize=ACADEMIC_STYLE['axes.labelsize'])
            ax2.set_ylabel(r'$q_3$', fontweight='bold', fontsize=ACADEMIC_STYLE['axes.labelsize'])
            ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2.grid(True, alpha=0.3)
        
        # Статистика
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        isotope = self.metadata.get('isotope', 'Unknown')
        energy = self.metadata.get('energy', 'N/A')
        n_traj = self.metadata.get('n_trajectories', 'N/A')
        dt = self.metadata.get('dt', 'N/A')
        
        # Составное ядро в LaTeX формате
        if '-' in isotope:
            element, mass_num = isotope.split('-')
            nucleus_latex = rf'$^{{{mass_num}}}{element}$'
        else:
            nucleus_latex = isotope
        
        stats_text = f"""Calculation Summary:
        
Nucleus: {nucleus_latex}
Energy: $E^* = {energy}$ MeV
Trajectories: $N = {n_traj}$
Time step: $\\Delta t = {dt}$ fm/c

Data points: {len(traj)}
"""
        
        # Добавляем информацию о пиках
        if mass_data and 'primary' in mass_data:
            primary_data = mass_data['primary']
            masses = primary_data['masses']
            yields = primary_data['yields']
            
            # Сортируем для поиска пиков
            sort_idx = np.argsort(masses)
            masses_sorted = masses[sort_idx]
            yields_sorted = yields[sort_idx]

            peak_indices = self._find_peaks(yields_sorted, masses_sorted)
            if peak_indices:
                total_mass = self.metadata.get('mass_number', None)
                fission_mode = self._classify_fission_mode(peak_indices, masses_sorted, total_mass)
                stats_text += f"\nFission mode: {fission_mode}"

                if len(peak_indices) == 1:
                    peak_mass = masses_sorted[peak_indices[0]]
                    stats_text += f"\nPeak: $A = {peak_mass:.0f}$"
                elif len(peak_indices) == 2:
                    light_mass = masses_sorted[peak_indices[0]]
                    heavy_mass = masses_sorted[peak_indices[1]]
                    stats_text += f"\nLight peak: $A_L = {light_mass:.0f}$"
                    stats_text += f"\nHeavy peak: $A_H = {heavy_mass:.0f}$"
                else:  # 3 пика
                    for i, peak_idx in enumerate(peak_indices):
                        peak_mass = masses_sorted[peak_idx]
                        if i == 0:
                            stats_text += f"\nLight peak: $A_L = {peak_mass:.0f}$"
                        elif i == 1:
                            stats_text += f"\nSymmetric: $A_S = {peak_mass:.0f}$"
                        else:
                            stats_text += f"\nHeavy peak: $A_H = {peak_mass:.0f}$"
        
        ax3.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                         edgecolor="black", alpha=0.9))
        
        # Заголовок сводки
        if '-' in isotope:
            element, mass_num = isotope.split('-')
            title_nucleus = rf'$^{{{mass_num}}}{element}$'
        else:
            title_nucleus = isotope
        
        plt.suptitle(f'FEL3D Analysis Summary - {title_nucleus}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(f"{save_path}_summary.png", dpi=300, bbox_inches='tight')
            print(f"✓ Summary saved: {save_path}_summary.png")
        
        plt.show()

###############################################################################
##################### ФУНКЦИИ ВЫСОКОГО УРОВНЯ ###############################
###############################################################################

def analyze_single_file(excel_file: str, output_dir: str = "plots", 
                       plot_options: Dict[str, bool] = None):
    """
    Анализ одного файла результатов
    
    Parameters:
    -----------
    excel_file : str
        Путь к Excel файлу
    output_dir : str
        Директория для сохранения графиков
    plot_options : dict
        Словарь с опциями построения графиков
    """
    if plot_options is None:
        plot_options = {
            'mass_distribution': True,
            'deformation': False,
            'temporal': False,
            'summary': False
        }
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {Path(excel_file).name}")
    print(f"{'='*60}")
    
    try:
        # Создание анализатора
        analyzer = FEL3DAnalyzer(excel_file)
        plotter = FEL3DPlotter(analyzer)
        
        # Создание директории для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Базовое имя для файлов
        base_name = os.path.join(output_dir, Path(excel_file).stem)
        
        # Построение графиков согласно опциям
        if plot_options.get('mass_distribution', True):
            plotter.plot_mass_distribution(base_name)
        
        if plot_options.get('deformation', False):
            plotter.plot_deformation_space(base_name)
        
        if plot_options.get('temporal', False):
            plotter.plot_temporal_evolution(base_name)
        
        if plot_options.get('summary', False):
            plotter.create_summary_report(base_name)
        
        print(f"✓ Analysis complete! Results in {output_dir}/")
        
    except Exception as e:
        print(f"✗ Error analyzing {excel_file}: {e}")


def analyze_directory(directory: str, pattern: str = "*.xlsx", 
                     output_dir: str = "plots", plot_options: Dict[str, bool] = None):
    """Анализ всех Excel файлов в директории"""
    from glob import glob
    
    search_path = os.path.join(directory, pattern)
    excel_files = glob(search_path)
    
    if not excel_files:
        print(f"No Excel files found in {directory}")
        return
    
    print(f"Found {len(excel_files)} Excel files for analysis")
    
    for excel_file in excel_files:
        try:
            file_output_dir = os.path.join(output_dir, Path(excel_file).stem)
            analyze_single_file(excel_file, file_output_dir, plot_options)
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
            continue


if __name__ == "__main__":
    # Простой интерфейс командной строки
    parser = argparse.ArgumentParser(description='FEL3D Mass Distribution Analysis')
    parser.add_argument('file', help='Excel file to analyze')
    parser.add_argument('-o', '--output', default='plots', help='Output directory')
    parser.add_argument('--all-plots', action='store_true', 
                       help='Generate all available plots')
    
    args = parser.parse_args()
    
    plot_opts = {
        'mass_distribution': True,
        'deformation': args.all_plots,
        'temporal': args.all_plots,
        'summary': args.all_plots
    }
    
    analyze_single_file(args.file, args.output, plot_opts)

#!/usr/bin/env python3
"""
Скрипт анализа результатов FEL3D
Фокус на массовом распределении фрагментов деления

Использование:
    python analyze_results.py                           # Интерактивный режим
    python analyze_results.py file.xlsx                 # Массовое распределение
    python analyze_results.py file.xlsx --all-plots     # Все графики
    python analyze_results.py -d Result/                # Анализ папки
    python analyze_results.py -l                        # Последние результаты
"""

import os
import sys
import argparse
from glob import glob
from datetime import datetime
from pathlib import Path

# Импорт модуля визуализации
try:
    from visualization import FEL3DAnalyzer, FEL3DPlotter, analyze_single_file, analyze_directory
except ImportError:
    print("Error: visualization.py module not found!")
    print("Make sure visualization.py is in the same directory.")
    sys.exit(1)

###############################################################################
##################### УТИЛИТЫ ПОИСКА ФАЙЛОВ ##################################
###############################################################################

def find_latest_results(base_dir="Result"):
    """
    Поиск самых свежих результатов
    
    Parameters:
    -----------
    base_dir : str
        Базовая директория для поиска
        
    Returns:
    --------
    list
        Список путей к последним файлам Excel
    """
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found!")
        return []
    
    # Поиск подпапок с датами
    date_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            try:
                # Попытка парсинга даты из имени папки
                datetime.strptime(item, "%d-%m-%y")
                date_dirs.append((item, item_path))
            except ValueError:
                continue
    
    if not date_dirs:
        # Поиск Excel файлов в корневой папке
        excel_files = glob(os.path.join(base_dir, "*.xlsx"))
        return sorted(excel_files, key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Сортировка по дате (новые первыми)
    date_dirs.sort(key=lambda x: datetime.strptime(x[0], "%d-%m-%y"), reverse=True)
    
    # Поиск Excel файлов в самой новой папке
    latest_dir = date_dirs[0][1]
    excel_files = glob(os.path.join(latest_dir, "*.xlsx"))
    
    return sorted(excel_files, key=lambda x: os.path.getmtime(x), reverse=True)


def find_all_results(base_dir="Result"):
    """
    Поиск всех файлов с результатами
    
    Parameters:
    -----------
    base_dir : str
        Базовая директория для поиска
        
    Returns:
    --------
    list
        Список всех найденных Excel файлов
    """
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found!")
        return []
    
    # Рекурсивный поиск всех Excel файлов
    excel_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.xlsx') and not file.startswith('~'):
                excel_files.append(os.path.join(root, file))
    
    # Сортировка по времени модификации (новые первыми)
    excel_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return excel_files


def select_files_interactive(files):
    """
    Интерактивный выбор файлов для анализа
    
    Parameters:
    -----------
    files : list
        Список доступных файлов
        
    Returns:
    --------
    tuple
        (список выбранных файлов, опции построения графиков)
    """
    if not files:
        print("No files found!")
        return [], {}
    
    print(f"\nFound {len(files)} result files:")
    print("-" * 80)
    
    for i, file in enumerate(files, 1):
        # Показываем только имя файла и дату модификации
        filename = Path(file).name
        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
        
        # Извлекаем основную информацию из имени файла
        parts = filename.lower().split('_')
        isotope = parts[0] if parts else 'unknown'
        energy = 'N/A'
        n_traj = 'N/A'
        
        for part in parts:
            if part.startswith('e(') and part.endswith(')'):
                energy = part[2:-1]
            elif part.startswith('n') and part[1:].isdigit():
                n_traj = part[1:]
        
        print(f"{i:2d}. {isotope.upper():<8} E={energy:<6} N={n_traj:<6} "
              f"{mod_time.strftime('%m-%d %H:%M')}")
    
    print("-" * 80)
    print("File selection:")
    print("  - Numbers (e.g., 1,3,5): Specific files")
    print("  - 'all': All files")
    print("  - 'latest': Most recent file")
    print("  - 'quit': Exit")
    
    choice = input("\nSelect files: ").strip().lower()
    
    if choice in ['quit', 'exit', 'q']:
        return [], {}
    elif choice == 'all':
        selected_files = files
    elif choice == 'latest':
        selected_files = [files[0]] if files else []
    else:
        # Парсинг номеров файлов
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_files = [files[i] for i in indices if 0 <= i < len(files)]
        except ValueError:
            print("Invalid input!")
            return [], {}
    
    if not selected_files:
        return [], {}
    
    # Выбор типа анализа
    print(f"\nSelected {len(selected_files)} files for analysis")
    print("\nAnalysis options:")
    print("1. Mass distribution only (default)")
    print("2. Mass distribution + deformation space")
    print("3. Mass distribution + temporal evolution")
    print("4. All plots")
    
    analysis_choice = input("Select analysis type (1-4, default=1): ").strip()
    
    # Настройка опций построения графиков
    plot_options = {
        'mass_distribution': True,
        'deformation': False,
        'temporal': False,
        'summary': False
    }
    
    if analysis_choice == '2':
        plot_options['deformation'] = True
    elif analysis_choice == '3':
        plot_options['temporal'] = True
    elif analysis_choice == '4':
        plot_options.update({
            'deformation': True,
            'temporal': True,
            'summary': True
        })
    
    return selected_files, plot_options


###############################################################################
##################### ИНТЕРАКТИВНЫЙ РЕЖИМ ####################################
###############################################################################

def interactive_mode():
    """Интерактивный режим работы"""
    print("=" * 70)
    print("           FEL3D Mass Distribution Analysis Tool")
    print("=" * 70)
    print("Focus: Fragment mass yield distributions with academic formatting")
    
    while True:
        print("\nMain Menu:")
        print("1. Analyze specific file")
        print("2. Analyze latest results")
        print("3. Browse and select files")
        print("4. Analyze all results in Result/ folder")
        print("5. Quick mass distribution (latest file)")
        print("6. Quit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            # Анализ конкретного файла
            file_path = input("Enter path to Excel file: ").strip()
            if os.path.exists(file_path):
                print("\nAnalysis options:")
                print("1. Mass distribution only")
                print("2. All plots")
                
                plot_choice = input("Select (1-2, default=1): ").strip()
                
                plot_options = {
                    'mass_distribution': True,
                    'deformation': plot_choice == '2',
                    'temporal': plot_choice == '2',
                    'summary': plot_choice == '2'
                }
                
                try:
                    analyze_single_file(file_path, plot_options=plot_options)
                except Exception as e:
                    print(f"Error analyzing file: {e}")
            else:
                print("File not found!")
                
        elif choice == '2':
            # Анализ последних результатов
            latest_files = find_latest_results()
            if latest_files:
                print(f"Found {len(latest_files)} latest result files")
                
                # Опции анализа
                print("\nAnalysis options:")
                print("1. Mass distribution only")
                print("2. All plots")
                
                plot_choice = input("Select (1-2, default=1): ").strip()
                
                plot_options = {
                    'mass_distribution': True,
                    'deformation': plot_choice == '2',
                    'temporal': plot_choice == '2',
                    'summary': plot_choice == '2'
                }
                
                for file in latest_files:
                    try:
                        output_dir = os.path.join("plots", Path(file).stem)
                        analyze_single_file(file, output_dir, plot_options)
                    except Exception as e:
                        print(f"Error analyzing {file}: {e}")
            else:
                print("No recent results found!")
                
        elif choice == '3':
            # Браузер файлов
            all_files = find_all_results()
            selected_files, plot_options = select_files_interactive(all_files)
            
            if selected_files:
                print(f"\nAnalyzing {len(selected_files)} files...")
                for file in selected_files:
                    try:
                        output_dir = os.path.join("plots", Path(file).stem)
                        analyze_single_file(file, output_dir, plot_options)
                    except Exception as e:
                        print(f"Error analyzing {file}: {e}")
                        
        elif choice == '4':
            # Анализ всех результатов
            all_files = find_all_results()
            if all_files:
                print(f"Found {len(all_files)} result files")
                confirm = input("Analyze all files? This may take time (y/N): ")
                
                if confirm.lower() in ['y', 'yes']:
                    plot_options = {'mass_distribution': True}
                    
                    for file in all_files:
                        try:
                            output_dir = os.path.join("plots", Path(file).stem)
                            analyze_single_file(file, output_dir, plot_options)
                        except Exception as e:
                            print(f"Error analyzing {file}: {e}")
            else:
                print("No results found!")
                
        elif choice == '5':
            # Быстрый анализ последнего файла
            latest_files = find_latest_results()
            if latest_files:
                latest_file = latest_files[0]
                print(f"Quick analysis: {Path(latest_file).name}")
                
                try:
                    plot_options = {'mass_distribution': True}
                    analyze_single_file(latest_file, "plots_quick", plot_options)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("No recent results found!")
                
        elif choice == '6':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice!")


###############################################################################
##################### ГЛАВНАЯ ФУНКЦИЯ ########################################
###############################################################################

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='FEL3D Fragment Mass Distribution Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive mode
  %(prog)s result.xlsx                  # Mass distribution only
  %(prog)s result.xlsx --all-plots      # All available plots
  %(prog)s -d Result/                   # Analyze directory
  %(prog)s -l                           # Analyze latest results
  %(prog)s -q                           # Quick analysis (latest file)
        """
    )
    
    parser.add_argument('file', nargs='?', help='Excel file to analyze')
    parser.add_argument('-d', '--directory', help='Directory to analyze')
    parser.add_argument('-l', '--latest', action='store_true', 
                       help='Analyze latest results')
    parser.add_argument('-q', '--quick', action='store_true',
                       help='Quick mass distribution (latest file)')
    parser.add_argument('-a', '--all-files', action='store_true',
                       help='Analyze all results in Result/ folder')
    parser.add_argument('-o', '--output', default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--all-plots', action='store_true',
                       help='Generate all available plots (not just mass distribution)')
    parser.add_argument('--deformation', action='store_true',
                       help='Include deformation space plot')
    parser.add_argument('--temporal', action='store_true',
                       help='Include temporal evolution plot')
    
    args = parser.parse_args()
    
    # Настройка опций построения графиков
    plot_options = {
        'mass_distribution': True,  # Всегда включено
        'deformation': args.deformation or args.all_plots,
        'temporal': args.temporal or args.all_plots,
        'summary': args.all_plots
    }
    
    # Обработка аргументов
    if args.file:
        # Анализ конкретного файла
        if os.path.exists(args.file):
            analyze_single_file(args.file, args.output, plot_options)
        else:
            print(f"File {args.file} not found!")
            sys.exit(1)
            
    elif args.directory:
        # Анализ директории
        if os.path.exists(args.directory):
            analyze_directory(args.directory, output_dir=args.output, 
                            plot_options=plot_options)
        else:
            print(f"Directory {args.directory} not found!")
            sys.exit(1)
            
    elif args.quick:
        # Быстрый анализ последнего файла
        latest_files = find_latest_results()
        if latest_files:
            latest_file = latest_files[0]
            print(f"Quick analysis: {Path(latest_file).name}")
            quick_options = {'mass_distribution': True}
            analyze_single_file(latest_file, "plots_quick", quick_options)
        else:
            print("No recent results found!")
            
    elif args.latest:
        # Анализ последних результатов
        latest_files = find_latest_results()
        if latest_files:
            print(f"Analyzing {len(latest_files)} latest result files...")
            for file in latest_files:
                file_output = os.path.join(args.output, Path(file).stem)
                analyze_single_file(file, file_output, plot_options)
        else:
            print("No recent results found!")
            
    elif args.all_files:
        # Анализ всех результатов
        all_files = find_all_results()
        if all_files:
            print(f"Analyzing {len(all_files)} result files...")
            confirm = input("This may take time. Continue? (y/N): ")
            if confirm.lower() in ['y', 'yes']:
                for file in all_files:
                    file_output = os.path.join(args.output, Path(file).stem)
                    analyze_single_file(file, file_output, plot_options)
        else:
            print("No results found!")
            
    else:
        # Интерактивный режим
        interactive_mode()


if __name__ == "__main__":
    main()

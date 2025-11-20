import os
import subprocess
import sys
from pathlib import Path

print("=" * 60)
print("АВТОМАТИЧЕСКАЯ ОБРАБОТКА ВСЕХ ИСПЫТУЕМЫХ")
print("=" * 60)

participant_folders = sorted([f for f in os.listdir('.') 
                              if os.path.isdir(f) and f.startswith('participant')])

if not participant_folders:
    print("\nОШИБКА: Не найдены папки с испытуемыми!")
    print("Ожидаются папки: participant1, participant2, participant3, и т.д.")
    sys.exit(1)

print(f"\nНайдено папок с испытуемыми: {len(participant_folders)}")
for folder in participant_folders:
    print(f"  - {folder}")

if not os.path.exists('process_eeg.py'):
    print("\nОШИБКА: Не найден файл process_eeg.py!")
    sys.exit(1)

print("\n" + "=" * 60)
print("НАЧАЛО ОБРАБОТКИ")
print("=" * 60)

successful = []
failed = []

for folder in participant_folders:
    print(f"\n{'='*60}")
    print(f"Обработка: {folder}")
    print(f"{'='*60}")
    
    folder_path = Path(folder)
    xlsx_files = list(folder_path.glob('*.xlsx'))
    
    if len(xlsx_files) < 3:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: В папке {folder} найдено только {len(xlsx_files)} xlsx файлов")
        print(f"  Ожидается минимум 3 файла (GO, NO-GO, behavioral)")
        failed.append(folder)
        continue
    
    original_dir = os.getcwd()
    os.chdir(folder)
    
    try:
        result = subprocess.run(
            [sys.executable, '../process_eeg.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            print(f"  OK: Успешно обработан: {folder}")
            successful.append(folder)
        else:
            print(f"  ERROR: Ошибка при обработке {folder}:")
            print(result.stderr)
            failed.append(folder)
    
    except Exception as e:
        print(f"  ERROR: {e}")
        failed.append(folder)
    
    finally:
        os.chdir(original_dir)

print("\n" + "=" * 60)
print("ИТОГОВЫЙ ОТЧЕТ")
print("=" * 60)
print(f"\nУспешно обработано: {len(successful)}")
for folder in successful:
    print(f"  OK: {folder}")

if failed:
    print(f"\nОшибки при обработке: {len(failed)}")
    for folder in failed:
        print(f"  ERROR: {folder}")

print("\n" + "=" * 60)
if len(successful) > 0:
    print("СЛЕДУЮЩИЙ ШАГ:")
    print("Запустите групповой анализ:")
    print("  python group_analysis.py")
    print("=" * 60)
else:
    print("ОШИБКА: Не удалось обработать ни одного испытуемого!")
    print("=" * 60)
    sys.exit(1)

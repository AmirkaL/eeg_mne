import pandas as pd
import numpy as np
import mne
from mne import create_info, EvokedArray
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys

plt.rcParams['figure.figsize'] = (12, 8)
mne.set_log_level('WARNING')

print("=" * 60)
print("ОБРАБОТКА ДАННЫХ ЭЭГ ИЗ ЭКСПЕРИМЕНТА GO/NO-GO")
print("=" * 60)

print("\n0. Поиск файлов в текущей папке...")

files_1 = glob.glob('*_1.xlsx')
files_2 = glob.glob('*_2.xlsx')
files_beh = glob.glob('*_beh.xlsx')

if not files_1:
    files_1 = glob.glob('*go*.xlsx') + glob.glob('*GO*.xlsx')
if not files_2:
    files_2 = glob.glob('*nogo*.xlsx') + glob.glob('*NOGO*.xlsx') + glob.glob('*no-go*.xlsx')
if not files_beh:
    files_beh = glob.glob('*beh*.xlsx') + glob.glob('*behavior*.xlsx')

if not files_1 or not files_2:
    all_xlsx = [f for f in glob.glob('*.xlsx') 
                if 'beh' not in f.lower() and 'behavior' not in f.lower()
                and 'statistics' not in f.lower() and 'result' not in f.lower()]
    if len(all_xlsx) >= 2:
        all_xlsx.sort()
        if not files_1:
            files_1 = [all_xlsx[0]]
        if not files_2:
            files_2 = [all_xlsx[1]]

if not files_1:
    print("ОШИБКА: Не найден файл с данными GO (ожидается *_1.xlsx или *go*.xlsx)")
    sys.exit(1)
if not files_2:
    print("ОШИБКА: Не найден файл с данными NO-GO (ожидается *_2.xlsx или *nogo*.xlsx)")
    sys.exit(1)
if not files_beh:
    print("ОШИБКА: Не найден файл с поведенческими данными (ожидается *_beh.xlsx)")
    sys.exit(1)

file_go = files_1[0]
file_nogo = files_2[0]
file_beh = files_beh[0]

print(f"   Найдены файлы:")
print(f"   - GO данные: {file_go}")
print(f"   - NO-GO данные: {file_nogo}")
print(f"   - Поведенческие данные: {file_beh}")

print("\n1. Чтение файлов...")
try:
    df1 = pd.read_excel(file_go)
    df2 = pd.read_excel(file_nogo)
    df_beh = pd.read_excel(file_beh)
    print("   Файлы успешно прочитаны")
except FileNotFoundError as e:
    print(f"   ОШИБКА: Файл не найден - {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ОШИБКА при чтении файлов: {e}")
    sys.exit(1)

df_beh.columns = df_beh.columns.str.strip()

print("\n2. Информация о стимулах:")
if 'Name' in df_beh.columns and 'Total' in df_beh.columns and 'Averaged' in df_beh.columns:
    print(df_beh[['Name', 'Total', 'Averaged', 'RT1']].to_string())
else:
    print("   Предупреждение: Не все ожидаемые колонки найдены в файле behavioral данных")
    print(f"   Доступные колонки: {df_beh.columns.tolist()}")

stimuli_data = []
for idx, row in df_beh.iterrows():
    name = str(row.get('Name', f'Stimul {idx+1}')).strip()
    total = row.get('Total', row.get('total', 0))
    averaged = row.get('Averaged', row.get('averaged', 0))
    if total > 0 or averaged > 0:
        stimuli_data.append({
            'name': name,
            'total': int(total) if pd.notna(total) else 0,
            'averaged': int(averaged) if pd.notna(averaged) else 0
        })

if len(stimuli_data) >= 2:
    stimuli_data.sort(key=lambda x: x['total'], reverse=True)
    
    go_stimulus = stimuli_data[0]['name']
    nogo_stimulus = stimuli_data[1]['name']
    go_trials = stimuli_data[0]['averaged']
    nogo_trials = stimuli_data[1]['averaged']
    
    print(f"\n   Автоматически определено:")
    print(f"   GO стимул: {go_stimulus} ({go_trials} усредненных эпох, всего {stimuli_data[0]['total']} trials)")
    print(f"   NO-GO стимул: {nogo_stimulus} ({nogo_trials} усредненных эпох, всего {stimuli_data[1]['total']} trials)")
elif len(stimuli_data) == 1:
    go_stimulus = stimuli_data[0]['name']
    nogo_stimulus = 'Stimul 2'
    go_trials = stimuli_data[0]['averaged']
    nogo_trials = 0
    print(f"\n   Найден только один стимул, используется как GO:")
    print(f"   GO стимул: {go_stimulus} ({go_trials} усредненных эпох)")
    print(f"   NO-GO стимул: {nogo_stimulus} (данные из файла {file_nogo})")
else:
    go_stimulus = 'Stimul 1'
    nogo_stimulus = 'Stimul 2'
    go_trials = 100
    nogo_trials = 100
    print(f"\n   Предупреждение: Не удалось определить стимулы из behavioral файла")
    print(f"   Используются значения по умолчанию")
    print(f"   GO стимул: {go_stimulus} ({go_trials} усредненных эпох)")
    print(f"   NO-GO стимул: {nogo_stimulus} ({nogo_trials} усредненных эпох)")

print("\n3. Подготовка данных для MNE...")

eeg_channels = [col for col in df1.columns if col not in ['Time (ms)', 'B1+']]
n_channels = len(eeg_channels)
n_samples = len(df1)

time_ms = df1['Time (ms)'].values
sampling_rate = 1000.0 / np.mean(np.diff(time_ms))
print(f"   Частота дискретизации: {sampling_rate:.2f} Гц")
print(f"   Количество каналов: {n_channels}")
print(f"   Количество отсчетов: {n_samples}")
print(f"   Временной диапазон: {time_ms[0]} - {time_ms[-1]} мс")

data_go = df1[eeg_channels].values.T
data_nogo = df2[eeg_channels].values.T

times = time_ms / 1000.0

montage = mne.channels.make_standard_montage('standard_1020')

info = create_info(
    ch_names=eeg_channels,
    sfreq=sampling_rate,
    ch_types='eeg'
)

info.set_montage(montage, match_case=False)

print("\n4. Создание Evoked объектов MNE...")

evoked_go = EvokedArray(
    data_go,
    info,
    tmin=times[0],
    comment='GO',
    nave=go_trials
)

evoked_nogo = EvokedArray(
    data_nogo,
    info,
    tmin=times[0],
    comment='NO-GO',
    nave=nogo_trials
)

print(f"   ERP GO: {evoked_go}")
print(f"   ERP NO-GO: {evoked_nogo}")

baseline = (times[0], 0.0)
evoked_go.apply_baseline(baseline)
evoked_nogo.apply_baseline(baseline)

print(f"   Применена базовая коррекция: {baseline[0]*1000:.0f} - {baseline[1]*1000:.0f} мс")

print("\n5. Создание визуализаций...")

available_times = [t for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] if times[0] <= t <= times[-1]]

if available_times:
    fig1 = evoked_go.plot_topomap(
        times=available_times,
        show=False
    )
    fig1.suptitle('ERP GO - Топографические карты', fontsize=14, fontweight='bold')
    plt.savefig('erp_go_topomap.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: erp_go_topomap.png")
    plt.close(fig1)

    fig2 = evoked_nogo.plot_topomap(
        times=available_times,
        show=False
    )
    fig2.suptitle('ERP NO-GO - Топографические карты', fontsize=14, fontweight='bold')
    plt.savefig('erp_nogo_topomap.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: erp_nogo_topomap.png")
    plt.close(fig2)

key_channels = ['Fz', 'Cz', 'Pz']
available_key_channels = [ch for ch in key_channels if ch in evoked_go.ch_names]

if len(available_key_channels) >= 2:
    n_plots = len(available_key_channels)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig3.suptitle('Сравнение ERP: GO vs NO-GO', fontsize=16, fontweight='bold')

    for idx, ch in enumerate(available_key_channels):
        ax = axes[idx]
        
        ch_idx = evoked_go.ch_names.index(ch)
        times_ms = evoked_go.times * 1000
        
        go_data = evoked_go.data[ch_idx, :] * 1e6
        nogo_data = evoked_nogo.data[ch_idx, :] * 1e6
        
        ax.plot(times_ms, go_data, label='GO', linewidth=2, color='blue')
        ax.plot(times_ms, nogo_data, label='NO-GO', linewidth=2, color='red')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Время (мс)', fontsize=12)
        ax.set_ylabel('Амплитуда (мкВ)', fontsize=12)
        ax.set_title(f'Канал {ch}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('erp_comparison.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: erp_comparison.png")
    plt.close(fig3)

evoked_diff = mne.combine_evoked([evoked_nogo, evoked_go], weights=[1, -1])

if available_times:
    fig4 = evoked_diff.plot_topomap(
        times=available_times,
        show=False
    )
    fig4.suptitle('Разностная волна (NO-GO - GO)', fontsize=14, fontweight='bold')
    plt.savefig('erp_difference_topomap.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: erp_difference_topomap.png")
    plt.close(fig4)

fig5, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
for ch_idx, ch_name in enumerate(evoked_go.ch_names):
    ax1.plot(evoked_go.times * 1000, evoked_go.data[ch_idx, :] * 1e6, 
             alpha=0.6, linewidth=0.8)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Время (мс)', fontsize=12)
ax1.set_ylabel('Амплитуда (мкВ)', fontsize=12)
ax1.set_title('ERP GO - Все каналы', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for ch_idx, ch_name in enumerate(evoked_nogo.ch_names):
    ax2.plot(evoked_nogo.times * 1000, evoked_nogo.data[ch_idx, :] * 1e6, 
             alpha=0.6, linewidth=0.8, color='red')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Время (мс)', fontsize=12)
ax2.set_ylabel('Амплитуда (мкВ)', fontsize=12)
ax2.set_title('ERP NO-GO - Все каналы', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('erp_butterfly.png', dpi=150, bbox_inches='tight')
print("   Сохранено: erp_butterfly.png")
plt.close(fig5)

print("\n6. Статистический анализ...")

difference = evoked_nogo.data - evoked_go.data

abs_diff = np.abs(difference)
max_ch_idx, max_time_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
max_diff_time = evoked_go.times[max_time_idx] * 1000
max_diff_channel = evoked_go.ch_names[max_ch_idx]

print(f"   Максимальная разность: {np.abs(difference).max() * 1e6:.2f} мкВ")
print(f"   Временная точка максимальной разности: {max_diff_time:.1f} мс")
print(f"   Канал с максимальной разностью: {max_diff_channel}")

time_window_n2 = (0.2, 0.3)
time_window_p3 = (0.3, 0.5)

if times[0] <= time_window_n2[0] and time_window_n2[1] <= times[-1]:
    n2_go = evoked_go.copy().crop(tmin=time_window_n2[0], tmax=time_window_n2[1])
    n2_nogo = evoked_nogo.copy().crop(tmin=time_window_n2[0], tmax=time_window_n2[1])
    
    n2_go_mean = np.mean(n2_go.data, axis=1) * 1e6
    n2_nogo_mean = np.mean(n2_nogo.data, axis=1) * 1e6
    
    print(f"\n   Компонент N2 (200-300 мс):")
    print(f"      GO средняя амплитуда: {np.mean(n2_go_mean):.2f} мкВ")
    print(f"      NO-GO средняя амплитуда: {np.mean(n2_nogo_mean):.2f} мкВ")
    print(f"      Разница: {np.mean(n2_nogo_mean - n2_go_mean):.2f} мкВ")
else:
    n2_go_mean = None
    n2_nogo_mean = None
    print(f"\n   Компонент N2: временное окно вне диапазона данных")

if times[0] <= time_window_p3[0] and time_window_p3[1] <= times[-1]:
    p3_go = evoked_go.copy().crop(tmin=time_window_p3[0], tmax=time_window_p3[1])
    p3_nogo = evoked_nogo.copy().crop(tmin=time_window_p3[0], tmax=time_window_p3[1])
    
    p3_go_mean = np.mean(p3_go.data, axis=1) * 1e6
    p3_nogo_mean = np.mean(p3_nogo.data, axis=1) * 1e6
    
    print(f"\n   Компонент P3 (300-500 мс):")
    print(f"      GO средняя амплитуда: {np.mean(p3_go_mean):.2f} мкВ")
    print(f"      NO-GO средняя амплитуда: {np.mean(p3_nogo_mean):.2f} мкВ")
    print(f"      Разница: {np.mean(p3_nogo_mean - p3_go_mean):.2f} мкВ")
else:
    p3_go_mean = None
    p3_nogo_mean = None
    print(f"\n   Компонент P3: временное окно вне диапазона данных")

print("\n7. Сохранение результатов...")

evoked_go.save('evoked_go-ave.fif', overwrite=True)
evoked_nogo.save('evoked_nogo-ave.fif', overwrite=True)
evoked_diff_fixed = evoked_diff.copy()
evoked_diff_fixed.nave = int(np.sqrt(evoked_go.nave * evoked_nogo.nave))
evoked_diff_fixed.save('evoked_difference-ave.fif', overwrite=True)

print("   Сохранено: evoked_go-ave.fif")
print("   Сохранено: evoked_nogo-ave.fif")
print("   Сохранено: evoked_difference-ave.fif")

print("\n   Создание Raw файлов для MNELab...")
from mne.io import RawArray

raw_go = RawArray(evoked_go.data, evoked_go.info, first_samp=0)
raw_go.save('go_raw.fif', overwrite=True)
print("   Сохранено: go_raw.fif (для MNELab)")

raw_nogo = RawArray(evoked_nogo.data, evoked_nogo.info, first_samp=0)
raw_nogo.save('nogo_raw.fif', overwrite=True)
print("   Сохранено: nogo_raw.fif (для MNELab)")

raw_diff = RawArray(evoked_diff.data, evoked_diff.info, first_samp=0)
raw_diff.save('difference_raw.fif', overwrite=True)
print("   Сохранено: difference_raw.fif (для MNELab)")

results_data = []
if n2_go_mean is not None:
    results_data.append({
        'Компонент': 'N2 (200-300 мс)',
        'GO средняя (мкВ)': np.mean(n2_go_mean),
        'NO-GO средняя (мкВ)': np.mean(n2_nogo_mean),
        'Разница (мкВ)': np.mean(n2_nogo_mean - n2_go_mean)
    })
if p3_go_mean is not None:
    results_data.append({
        'Компонент': 'P3 (300-500 мс)',
        'GO средняя (мкВ)': np.mean(p3_go_mean),
        'NO-GO средняя (мкВ)': np.mean(p3_nogo_mean),
        'Разница (мкВ)': np.mean(p3_nogo_mean - p3_go_mean)
    })

if results_data:
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('erp_statistics.csv', index=False, encoding='utf-8-sig')
    print("   Сохранено: erp_statistics.csv")

print("\n" + "=" * 60)
print("ИТОГОВЫЙ ОТЧЕТ")
print("=" * 60)
print(f"\nОбработано данных:")
print(f"  - GO стимул: {go_trials} усредненных эпох")
print(f"  - NO-GO стимул: {nogo_trials} усредненных эпох")
print(f"\nПараметры обработки:")
print(f"  - Частота дискретизации: {sampling_rate:.2f} Гц")
print(f"  - Базовый период: {baseline[0]*1000:.0f} - {baseline[1]*1000:.0f} мс")
print(f"  - Временной диапазон анализа: {times[0]*1000:.0f} - {times[-1]*1000:.0f} мс")
print(f"  - Количество каналов: {n_channels}")
print(f"\nСозданные файлы:")
print(f"  - erp_go_topomap.png - Топографические карты GO")
print(f"  - erp_nogo_topomap.png - Топографические карты NO-GO")
print(f"  - erp_comparison.png - Сравнение GO и NO-GO")
print(f"  - erp_difference_topomap.png - Разностная волна")
print(f"  - erp_butterfly.png - Все каналы (butterfly plot)")
print(f"  - evoked_go-ave.fif - Данные ERP GO (MNE формат, Evoked)")
print(f"  - evoked_nogo-ave.fif - Данные ERP NO-GO (MNE формат, Evoked)")
print(f"  - evoked_difference-ave.fif - Разностная волна (MNE формат, Evoked)")
print(f"  - go_raw.fif - Данные GO в формате Raw (для MNELab)")
print(f"  - nogo_raw.fif - Данные NO-GO в формате Raw (для MNELab)")
print(f"  - difference_raw.fif - Разностная волна в формате Raw (для MNELab)")
if results_data:
    print(f"  - erp_statistics.csv - Статистика компонентов")
print("\n" + "=" * 60)
print("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
print("=" * 60)

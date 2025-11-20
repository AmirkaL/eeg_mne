import pandas as pd
import numpy as np
import mne
from mne import read_evokeds
import matplotlib.pyplot as plt
import glob
import os
import sys

plt.rcParams['figure.figsize'] = (14, 10)
mne.set_log_level('WARNING')

print("=" * 60)
print("ГРУППОВОЙ АНАЛИЗ ДАННЫХ ЭЭГ (GRAND AVERAGE)")
print("=" * 60)

print("\n1. Поиск файлов испытуемых...")

go_files = []
nogo_files = []

for folder in sorted(glob.glob('participant*')):
    if os.path.isdir(folder):
        go_file = os.path.join(folder, 'evoked_go-ave.fif')
        nogo_file = os.path.join(folder, 'evoked_nogo-ave.fif')
        
        if os.path.exists(go_file):
            go_files.append(go_file)
        if os.path.exists(nogo_file):
            nogo_files.append(nogo_file)

print(f"   Найдено файлов GO: {len(go_files)}")
print(f"   Найдено файлов NO-GO: {len(nogo_files)}")

if len(go_files) == 0 or len(nogo_files) == 0:
    print("\n   ОШИБКА: Не найдены evoked файлы!")
    print("   Убедитесь, что вы обработали всех испытуемых:")
    print("   python process_all_participants.py")
    sys.exit(1)

print("\n   Найденные файлы:")
for i, (go_file, nogo_file) in enumerate(zip(go_files, nogo_files), 1):
    print(f"   {i}. {os.path.dirname(go_file)}")

print("\n2. Загрузка данных испытуемых...")

evoked_go_list = []
evoked_nogo_list = []

for go_file in go_files:
    try:
        evoked = read_evokeds(go_file, verbose=False)
        if isinstance(evoked, list):
            evoked = evoked[0]
        evoked_go_list.append(evoked)
        print(f"   Загружен: {os.path.basename(os.path.dirname(go_file))}")
    except Exception as e:
        print(f"   ОШИБКА при загрузке {go_file}: {e}")

for nogo_file in nogo_files:
    try:
        evoked = read_evokeds(nogo_file, verbose=False)
        if isinstance(evoked, list):
            evoked = evoked[0]
        evoked_nogo_list.append(evoked)
    except Exception as e:
        print(f"   ОШИБКА при загрузке {nogo_file}: {e}")

if len(evoked_go_list) == 0 or len(evoked_nogo_list) == 0:
    print("   ОШИБКА: Не удалось загрузить данные!")
    sys.exit(1)

print(f"\n   Успешно загружено: {len(evoked_go_list)} испытуемых")

print("\n3. Создание группового усреднения (Grand Average)...")

grand_avg_go = mne.grand_average(evoked_go_list)
grand_avg_nogo = mne.grand_average(evoked_nogo_list)

grand_avg_diff = mne.combine_evoked([grand_avg_nogo, grand_avg_go], weights=[1, -1])

print(f"   Grand Average GO: {grand_avg_go}")
print(f"   Grand Average NO-GO: {grand_avg_nogo}")
print(f"   Количество испытуемых: {len(evoked_go_list)}")

print("\n4. Создание групповых визуализаций...")

available_times = [t for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] 
                   if grand_avg_go.times[0] <= t <= grand_avg_go.times[-1]]

if available_times:
    fig1 = grand_avg_go.plot_topomap(times=available_times, show=False)
    fig1.suptitle(f'Grand Average GO (N={len(evoked_go_list)})', 
                  fontsize=14, fontweight='bold')
    plt.savefig('grand_average_go_topomap.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: grand_average_go_topomap.png")
    plt.close(fig1)

    fig2 = grand_avg_nogo.plot_topomap(times=available_times, show=False)
    fig2.suptitle(f'Grand Average NO-GO (N={len(evoked_nogo_list)})', 
                  fontsize=14, fontweight='bold')
    plt.savefig('grand_average_nogo_topomap.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: grand_average_nogo_topomap.png")
    plt.close(fig2)

    fig3 = grand_avg_diff.plot_topomap(times=available_times, show=False)
    fig3.suptitle(f'Grand Average Difference (NO-GO - GO, N={len(evoked_go_list)})', 
                  fontsize=14, fontweight='bold')
    plt.savefig('grand_average_difference_topomap.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: grand_average_difference_topomap.png")
    plt.close(fig3)

key_channels = ['Fz', 'Cz', 'Pz']
available_key_channels = [ch for ch in key_channels if ch in grand_avg_go.ch_names]

if len(available_key_channels) >= 2:
    n_plots = len(available_key_channels)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig4, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig4.suptitle(f'Grand Average: GO vs NO-GO (N={len(evoked_go_list)})', 
                  fontsize=16, fontweight='bold')

    for idx, ch in enumerate(available_key_channels):
        ax = axes[idx]
        ch_idx = grand_avg_go.ch_names.index(ch)
        times_ms = grand_avg_go.times * 1000
        
        go_data = grand_avg_go.data[ch_idx, :] * 1e6
        nogo_data = grand_avg_nogo.data[ch_idx, :] * 1e6
        
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
    plt.savefig('grand_average_comparison.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: grand_average_comparison.png")
    plt.close(fig4)

fig5, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
for ch_idx, ch_name in enumerate(grand_avg_go.ch_names):
    ax1.plot(grand_avg_go.times * 1000, grand_avg_go.data[ch_idx, :] * 1e6, 
             alpha=0.6, linewidth=0.8)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Время (мс)', fontsize=12)
ax1.set_ylabel('Амплитуда (мкВ)', fontsize=12)
ax1.set_title(f'Grand Average GO (N={len(evoked_go_list)})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for ch_idx, ch_name in enumerate(grand_avg_nogo.ch_names):
    ax2.plot(grand_avg_nogo.times * 1000, grand_avg_nogo.data[ch_idx, :] * 1e6, 
             alpha=0.6, linewidth=0.8, color='red')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Время (мс)', fontsize=12)
ax2.set_ylabel('Амплитуда (мкВ)', fontsize=12)
ax2.set_title(f'Grand Average NO-GO (N={len(evoked_nogo_list)})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grand_average_butterfly.png', dpi=150, bbox_inches='tight')
print("   Сохранено: grand_average_butterfly.png")
plt.close(fig5)

print("\n5. Групповая статистика...")

time_window_n2 = (0.2, 0.3)
time_window_p3 = (0.3, 0.5)

n2_go_amplitudes = []
n2_nogo_amplitudes = []
p3_go_amplitudes = []
p3_nogo_amplitudes = []

for evoked_go, evoked_nogo in zip(evoked_go_list, evoked_nogo_list):
    if (evoked_go.times[0] <= time_window_n2[0] and 
        time_window_n2[1] <= evoked_go.times[-1]):
        n2_go = evoked_go.copy().crop(tmin=time_window_n2[0], tmax=time_window_n2[1])
        n2_nogo = evoked_nogo.copy().crop(tmin=time_window_n2[0], tmax=time_window_n2[1])
        n2_go_amplitudes.append(np.mean(n2_go.data) * 1e6)
        n2_nogo_amplitudes.append(np.mean(n2_nogo.data) * 1e6)
    
    if (evoked_go.times[0] <= time_window_p3[0] and 
        time_window_p3[1] <= evoked_go.times[-1]):
        p3_go = evoked_go.copy().crop(tmin=time_window_p3[0], tmax=time_window_p3[1])
        p3_nogo = evoked_nogo.copy().crop(tmin=time_window_p3[0], tmax=time_window_p3[1])
        p3_go_amplitudes.append(np.mean(p3_go.data) * 1e6)
        p3_nogo_amplitudes.append(np.mean(p3_nogo.data) * 1e6)

stats_data = []

if n2_go_amplitudes:
    n2_go_arr = np.array(n2_go_amplitudes)
    n2_nogo_arr = np.array(n2_nogo_amplitudes)
    stats_data.append({
        'Компонент': 'N2 (200-300 мс)',
        'GO средняя (мкВ)': np.mean(n2_go_arr),
        'GO ст.откл. (мкВ)': np.std(n2_go_arr),
        'NO-GO средняя (мкВ)': np.mean(n2_nogo_arr),
        'NO-GO ст.откл. (мкВ)': np.std(n2_nogo_arr),
        'Разница (мкВ)': np.mean(n2_nogo_arr - n2_go_arr),
        'N испытуемых': len(n2_go_amplitudes)
    })

if p3_go_amplitudes:
    p3_go_arr = np.array(p3_go_amplitudes)
    p3_nogo_arr = np.array(p3_nogo_amplitudes)
    stats_data.append({
        'Компонент': 'P3 (300-500 мс)',
        'GO средняя (мкВ)': np.mean(p3_go_arr),
        'GO ст.откл. (мкВ)': np.std(p3_go_arr),
        'NO-GO средняя (мкВ)': np.mean(p3_nogo_arr),
        'NO-GO ст.откл. (мкВ)': np.std(p3_nogo_arr),
        'Разница (мкВ)': np.mean(p3_nogo_arr - p3_go_arr),
        'N испытуемых': len(p3_go_amplitudes)
    })

if stats_data:
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv('grand_average_statistics.csv', index=False, encoding='utf-8-sig')
    print("   Сохранено: grand_average_statistics.csv")
    print("\n   Групповая статистика:")
    print(stats_df.to_string(index=False))

print("\n6. Сохранение Grand Average...")

grand_avg_go.save('grand_average_go-ave.fif', overwrite=True)
grand_avg_nogo.save('grand_average_nogo-ave.fif', overwrite=True)
grand_avg_diff.save('grand_average_difference-ave.fif', overwrite=True)

print("   Сохранено: grand_average_go-ave.fif")
print("   Сохранено: grand_average_nogo-ave.fif")
print("   Сохранено: grand_average_difference-ave.fif")

print("\n" + "=" * 60)
print("ГРУППОВОЙ АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 60)
print(f"\nОбработано испытуемых: {len(evoked_go_list)}")
print(f"\nСозданные файлы:")
print(f"  - grand_average_go_topomap.png")
print(f"  - grand_average_nogo_topomap.png")
print(f"  - grand_average_difference_topomap.png")
print(f"  - grand_average_comparison.png")
print(f"  - grand_average_butterfly.png")
print(f"  - grand_average_go-ave.fif")
print(f"  - grand_average_nogo-ave.fif")
print(f"  - grand_average_difference-ave.fif")
print(f"  - grand_average_statistics.csv")
print("\n" + "=" * 60)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm



#################################    Для настройки порога яркости   #####################################


def plot_events(events, info=None):
    """

    events: список событий [(t_1, x_1, y_1, p_1), ..., (t_n, x_n, y_n, p_n)]
    dt: время между кадрами

    Строит:
        1) гистограмму по времени (сколько событий в каждом промежутке)
        2) диаграммы рассеяния (t vs x) и (t vs y) с окраской по полярности:
            -- синий: пиксель стал ярче (полярность p=1)
            -- красный: пиксель стал темнее (полярность p=0)

    """
    if not events:
        return

    times = [e[0] for e in events]
    xs = [int(e[1]) for e in events]
    ys = [int(e[2]) for e in events]
    ps = [e[3] for e in events]

    t_min = np.floor(min(times))
    t_max = np.ceil(max(times))

    bin_width = 1.0  # Шаг 1 мс
    bins = np.arange(t_min, t_max + bin_width, bin_width)

    plt.figure(figsize=(12, 4))

    # Гистограмма по времени
    plt.subplot(1, 3, 1)
    plt.hist(times, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Распределение событий по времени")
    plt.xlabel("Время (мс)")
    plt.ylabel("Количество событий")

    # t vs x
    plt.subplot(1, 3, 2)
    plt.scatter(times, xs, c=ps, cmap='bwr', s=5, alpha=0.7)
    plt.xlabel("Время (мс)")
    plt.ylabel("X")
    plt.title("t vs X, цвет = полярность")

    # t vs y
    plt.subplot(1, 3, 3)
    plt.scatter(times, ys, c=ps, cmap='bwr', s=5, alpha=0.7)
    plt.xlabel("Время (мс)")
    plt.ylabel("Y")
    plt.title("t vs Y, цвет = полярность")
    plt.gca().invert_yaxis()

    if info is not None:
        plt.suptitle(f"Направление: {info}", fontsize=14)

    plt.tight_layout()
    plt.show()



def plot_events_3d(events, info=None):
    if not events:
        return

    times = [e[0] for e in events]
    xs = [e[1] for e in events]
    ys = [e[2] for e in events]
    ps = [e[3] for e in events] 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Используем полярность в качестве входа для colormap, например 'bwr' или другую
    scatter = ax.scatter(xs, ys, times, c=ps, cmap='bwr', s=5, alpha=0.7)

    ax.set_xlabel('X (гориз. пиксель)')
    ax.set_ylabel('Y (верт. пиксель)')
    ax.invert_yaxis()
    ax.set_zlabel('Время (мс)')
    ax.invert_yaxis()

    # Легенда для полярности: создадим цветную шкалу
    cb = fig.colorbar(scatter, ax=ax, pad=0.1)
    cb.set_label('Полярность (0=off, 1=on)')

    if info is not None:
        ax.set_title(str(info), pad=15)

    plt.show()


############################### Проверка датасета #####################################


def show_trajectory(sample):
    frames = sample.get("frames", [])
    if not frames:
        return

    n_plot = len(frames)

    fig, axes = plt.subplots(1, n_plot, figsize=(12, 2.5))
    if n_plot == 1:
        axes = [axes]  # чтобы иметь единый подход при цикле

    for i in range(n_plot):
        ax = axes[i]
        ax.imshow(frames[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Frame {i}")
        ax.axis("off")

    meta_info = []
    if "direction" in sample:
        meta_info.append(f"dir: {sample['direction']}")
    if "speed" in sample:
        meta_info.append(f"speed: {sample['speed']}")
    if "style" in sample:
        meta_info.append(f"style: {sample['style']}")
    if "start_pos" in sample:
        meta_info.append(f"start_pos: {sample['start_pos']}")

    if meta_info:
        fig.suptitle(", ".join(meta_info), fontsize=10)

    plt.tight_layout()
    plt.show()


#################################    Результат обучения   #####################################

# Количество спайков по направлениям (гистограмма)
def plot_direction_hist(spikes_count_by_dir, directions):
    count_neurons = len(spikes_count_by_dir[directions[0]])
    all_heights = []
    all_labels = []
    all_colors = []

    for i in range(count_neurons):
        color = matplotlib.cm.tab10(i % 10) 
        for d in directions:
            c = spikes_count_by_dir[d][i]
            all_heights.append(c)
            all_labels.append(f"N{i}-{d}")
            all_colors.append(color)

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(all_heights)), all_heights, color=all_colors)
    plt.xticks(range(len(all_heights)), all_labels, rotation=90)
    plt.title("Количество спайков нейронов по направлениям")
    plt.tight_layout()
    plt.show()




def plot_direction_heatmap(spikes_count_by_dir, directions, count_neurons):
    """
    
    Строит тепловую карту, показывающую, как часто каждый нейрон реагирует на каждое направление.

    """
    # Собираем матрицу shape=(count_neurons, len(directions))
    matrix = np.zeros((count_neurons, len(directions)), dtype=int)
    for j, d in enumerate(directions):
        matrix[:, j] = spikes_count_by_dir[d]

    plt.figure(figsize=(10, 0.5 * count_neurons + 2))
    plt.imshow(matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Spike count')

    plt.xticks(ticks=range(len(directions)), labels=[str(d) for d in directions], rotation=45)
    plt.yticks(ticks=range(count_neurons), labels=[f"N{i}" for i in range(count_neurons)])
    plt.xlabel("Direction")
    plt.ylabel("Neuron ID")
    plt.title("Neuron selectivity heatmap")
    plt.tight_layout()
    plt.show()

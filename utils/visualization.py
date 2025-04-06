import matplotlib.pyplot as plt
import numpy as np



#################################    Для настройки порога яркости   #####################################


def plot_events(events, dt=10):
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

    t_min = min(times)
    t_max = max(times)

    bin_width = dt
    bins = np.arange(t_min, t_max + bin_width, bin_width)


    plt.figure()

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

    plt.tight_layout()
    plt.show()






import numpy as np
import genetic.ga_config as ga
from core import global_config as cfg
from core.input_layer import (
    init_event_generator, 
    generate_events
)
from core.hidden_layer import (
    init_hidden_layer,
    reset_hidden_layer,
    hidden_layer_step
)


# Константа для вычислений
LOG8 = np.log(8)



# Обновление значений гиперпараметров сети
def _set_params(params):
    cfg.TAU_LEAK = params["TAU_LEAK"]
    cfg.I_THRES = params["I_THRES"]
    cfg.T_REF = params["T_REF"]
    cfg.T_INHIBIT = params["T_INHIBIT"]
    cfg.ALPHA_PLUS = params["ALPHA_PLUS"]
    cfg.ALPHA_MINUS = params["ALPHA_MINUS"]
    cfg.BETA_PLUS = params["BETA_PLUS"]
    cfg.BETA_MINUS = params["BETA_MINUS"]
    cfg.T_LTP = params["T_LTP"]
    cfg.W_INIT_MEAN = params["W_INIT_MEAN"]
    cfg.W_INIT_STD = params["W_INIT_STD"]
    cfg.W_MIN = params["W_MIN"]
    cfg.W_MAX = params["W_MAX"]
    cfg.W_RANGE = cfg.W_MAX - cfg.W_MIN



# Прогон алгоритма на наборе гиперпараметров params и оценка селективности скрытого слоя
def evaluate_selectivity(
        params,                 # словарь гиперпараметров сети
        distr_penalty=0.3,      # вес штрафа за неравномерное распределение нейронов по направлениям
        dataset=None            # датасет (словарь)
):
    np.random.seed(hash(frozenset(params.items())) & 0xFFFFFFFF)

    # Запоминаем количество кадров в одном примере датасета
    num_frames = len(dataset[0]["frames"])

    # Обновляем глобальные константы
    _set_params(params)

    # Инициализируем скрытый слой
    hidden = init_hidden_layer()

    # Заводим статистику спайков по направлениям 
    # (строки - нейроны, столбцы - направления; ячейка - количество спайков)
    spike_matrix = np.zeros((cfg.COUNT_NEURONS, 8), dtype=np.int32)

    for _ in range(ga.EPOCHS):
        dataset_local = np.random.permutation(dataset)
        spike_matrix[:, :] = 0
        # Прогоняем алгоритм на каждом примере (последовательность кадров) из датасета
        for sample in dataset_local:
            # Инициализируем генератор событий
            ev_gen = init_event_generator()
            # Запоминаем первый кадр
            prev_frame = sample["frames"][0]
            # Устанавливаем начальное время симуляции
            prev_t = 0.0
            # Обрабатываем кадры, начиная со второго
            for frame_i in range(1, num_frames):
                # Берем следующий кадр
                new_frame = sample["frames"][frame_i]
                # Фиксируем время поступления нового кадра
                new_t = frame_i * cfg.FRAME_DT_MS

                # Генерируем события между двумя соседними кадрами
                events = generate_events(
                    state=ev_gen,
                    old_frame=prev_frame,
                    new_frame=new_frame,
                    prev_t=prev_t,
                    new_t=new_t
                )

                # Для адаптации величины сигнала по количеству событий
                norm_factor = min(1.0, ga.AVERAGE_EV_PER_FRAME/(len(events) + 1e-12))

                # Передаем события в скрытый слой
                for ev in events:
                    hidden_layer_step(
                        state=hidden,
                        event=ev,
                        train=True, # обучение
                        norm_factor=norm_factor          
                    )

                # Обновляем кадр
                prev_frame, prev_t = new_frame, new_t

            # Сопоставляем текущее направление движения его номеру
            dir_ = sample["direction"]
            dir_idx = ga.DIR2IDX[tuple(dir_)]
            spikes_this_sample = np.zeros(cfg.COUNT_NEURONS, dtype=np.bool_)

            # Считаем количество спайков за это направление
            for (_, neuron_idx) in hidden["spikes"]:
                spike_matrix[neuron_idx, dir_idx] += 1
                spikes_this_sample[neuron_idx] += 1

            # Для слишком активных нейронов повышаем порог
            if_overactive = spikes_this_sample >= ga.MAX_SPIKES_PER_SAMPLE
            hidden["thresh"][if_overactive] *= ga.HOMEO_UP
            # Для неактивных понижаем 
            # (неактивными считаются те, которые молчат не менее NUM_INACTIVE_SAMPLES примеров подряд)
            if_inactivity = spikes_this_sample == 0
            hidden["inactivity"][if_inactivity] += 1
            hidden["inactivity"][~if_inactivity] = 0
            need_down = hidden["inactivity"] >= ga.NUM_INACTIVE_SAMPLES
            hidden["thresh"][need_down] *= ga.HOMEO_DOWN
            # Ограничиваем в допустимых диапазонах
            hidden["thresh"] = np.clip(hidden["thresh"], 0.1 * cfg.I_THRES, 5 * cfg.I_THRES)

            # Сбрасываем состояние нейронов скрытого слоя
            reset_hidden_layer(hidden)

    # Считаем общее количество спайков для каждого нейрона за весь датасет
    spikes_per_neuron = spike_matrix.sum(axis=1, dtype=np.float32)
    # Защита от деления на ноль
    spikes_per_neuron += 1e-12
    # Вычисляем предпочтения нейронов (долю голосов каждого нейрона за каждое из направлений)
    # p - двумерная матрица, где p[i, j] хранит долю спайков нейрона i за направление j
    p = spike_matrix / spikes_per_neuron[:, None]
    # Логарифм по основанию 8 (8 направлений движения)
    log_p = np.log(p + 1e-12) / LOG8
    # Энтропия Шеннона для оценки селективности нейронов
    H = -(p * log_p).sum(axis=1)
    # Для неактивных нейронов ставим высокую энтропию
    H[spikes_per_neuron < ga.MIN_SPIKES_FOR_ACTIVE] = 1.0 
    # Среднее значение энтропии         
    H_mean = H.mean()

    # Для каждого направления считаем, какое количество нейронов выбрало его как "любимое"
    counts = np.zeros(8, dtype=np.float32)
    for neuron_idx in range(cfg.COUNT_NEURONS):
        # Выбираем индекс направления с максимальным числом спайков
        pref = np.argmax(spike_matrix[neuron_idx])
        # Увеличиваем счетчик для этого направления
        counts[pref] += 1

    # Доля нейронов на каждое направление
    neuron_frac = counts / cfg.COUNT_NEURONS
    # В идеале на каждое направление должно быть одинаковое количество нейронов
    # Вычитаем 1/8, чтобы оценить, на сколько полученное распределение отличается от идеала
    neuron_frac -= 0.125
    # Вычисляем среднее значение отклонения
    average_dev = np.abs(neuron_frac).mean()

    # Оцениваем качество гиперпараметров
    # Чем больше значение, тем хуже селективность
    anti_selectivity_score = H_mean + distr_penalty * average_dev 

    return anti_selectivity_score, spike_matrix

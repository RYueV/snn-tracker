import numpy as np
from core.neuron import process_lif_event
from core.learning import update_weights_stdp, apply_inhibition

"""

Скрытый слой: события -> распределение спайков по направлениям.
Каждый нейрон связан со всеми входами.

"""

def init_hidden_layer(
        count_neurons,          # количество нейронов в скрытом слое
        input_shape=(28,28),    # размер входного изображения (кадра)
        w_init_mean=260.0,      # среднее значение начальных весов
        w_init_std=52.0,        # разброс относительно среднего
        w_min=100.0,            # минимально возможный вес
        w_max=400.0             # максимально возможный вес
):
    # Количество входов (на каждый пиксель 2 состояния)
    input_size = input_shape[0] * input_shape[1] * 2 
    # Матрица весов (строки - нейроны, столбцы - входы)
    weights = np.random.normal(w_init_mean, w_init_std, size=(count_neurons, input_size))
    weights = np.clip(weights, w_min, w_max)

    # Словарь гиперпараметров сети
    state = {
        "count_neurons": count_neurons,
        "input_size": input_size,
        "weights": weights,
        # Текущее значение потенциала для каждого нейрона сети
        "membrane_potentials": np.zeros(count_neurons, dtype=np.float32),
        # Время последней активации для каждого входа
        "last_input_times": np.zeros(input_size, dtype=np.float32),
        # Время последнего спайка для каждого нейрона
        "last_spike_times": np.full(count_neurons, -np.inf, dtype=np.float32),
        # Время окончания ингибирования каждого нейрона
        "inhibited_until": np.zeros(count_neurons, dtype=np.float32),
        # Массив спайков: (момент времени, номер нейрона)
        "spikes": []
    }
    return state



# Сброс настроек скрытого слоя
def reset_hidden_layer(state):
    state["membrane_potentials"].fill(0.0)
    state["last_input_times"].fill(0.0)
    state["last_spike_times"].fill(-np.inf)
    state["inhibited_until"].fill(0.0)
    state["spikes"].clear()



# Обработка одного входного события в скрытом слое
def hidden_layer_step(
        state,              # словарь гиперпараметров сети
        event,              # событие из input_layer
        train=True          # если False, веса зафиксированы
):
    # Время события, координаты пикселя и полярность
    t, x, y, p = event
    # Превращаем двумерные координаты в индекс входа
    input_id = 2 * (y * 28 + x) + p
    # Обновляем время последней активации входа
    state["last_input_times"][input_id] = t

    # Вход связан со всеми нейронами, поэтому обработка события выполняется для каждого нейрона
    count_neurons = state["count_neurons"]
    for neuron_id in range(count_neurons):
        # Подаем событие нейрону
        spiked, new_u = process_lif_event(
            input_time=t,                                           # время подачи входного сигнала
            u=state["membrane_potentials"][neuron_id],              # текущий мембранный потенциал нейрона
            last_input=state["last_input_times"][input_id],         # время последнего входного сигнала
            last_spike=state["last_spike_times"][neuron_id],        # время последнего спайка нейрона
            inhibited_until=state["inhibited_until"][neuron_id],    # время, до которого нейрон "подавлен"
            weight=state["weights"][neuron_id][input_id]            # вес связи нейрона со входом
        )
        # Обновляем значение потенциала нейрона
        state["membrane_potentials"][neuron_id] = new_u

        # Если при обработке события был спайк
        if spiked:
            # Фиксируем время, когда сработал нейрон и его индекс
            state["spikes"].append((t, neuron_id))
            # Обновляем время последнего спайка
            state["last_spike_times"][neuron_id] = t
            # Ингибирование всех нейронов, кроме сработавшего
            apply_inhibition(
                inhibited_until_array=state["inhibited_until"],
                t=t,
                spiking_index=neuron_id
            )
            # Если сеть обучается, то обновляем веса по правилу STDP
            if train:
                update_weights_stdp(
                    neuron_index=neuron_id,                     # индекс нейрона
                    t_post=t,                                   # время, когда нейрон сработал
                    weights=state["weights"],                   # матрица весов
                    last_input_times=state["last_input_times"]  # вектор времен активации входов
                )

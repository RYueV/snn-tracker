import numpy as np
import core.hidden_config as h_conf
from core.learning import update_weights_stdp, apply_inhibition


"""

Скрытый слой: события -> распределение спайков по направлениям.
Каждый нейрон связан со всеми входами.

"""


# Инициализация скрытого слоя
def init_hidden_layer():
    # Количество входов (на каждый пиксель 2 состояния)
    input_size = h_conf.IMAGE_HEIGHT * h_conf.IMAGE_WIDTH * 2
    # Инициализация весов
    weights = np.clip(
        np.random.normal(
            h_conf.W_INIT_MEAN, h_conf.W_INIT_STD, (h_conf.COUNT_NEURONS, input_size)
        ),
        h_conf.W_MIN, h_conf.W_MAX
    ).astype(np.float32)

    return {
        # Текущее значение потенциала для каждого нейрона сети
        "membrane_potentials": np.zeros(h_conf.COUNT_NEURONS, np.float32),
        # Время последней активации каждого входа
        "last_input_times": np.zeros(input_size, np.float32),
        # Время последнего обновления потенциала каждого нейрона сети
        "last_update": np.zeros(h_conf.COUNT_NEURONS, np.float32),
        # Время последнего спайка каждого нейрона сети
        "last_spike": np.full(h_conf.COUNT_NEURONS, -np.inf, np.float32),
        # Время окончания периода ингибирования для каждого нейрона
        "inhibited_until": np.zeros(h_conf.COUNT_NEURONS, np.float32),
        # Матрица весов
        "weights": weights,
        # Текущее время симуляции
        "current_ms": None,
        # Накопленный вклад событий за текущую миллисекунду.
        "batch_sum": np.zeros(h_conf.COUNT_NEURONS, np.float32),
        # Массив спайков: (момент времени, номер нейрона)
        "spikes": []
    }




# Сброс настроек скрытого слоя
def reset_hidden_layer(state):
    state["membrane_potentials"].fill(0.0)
    state["last_update"].fill(0.0)
    state["last_spike"].fill(-np.inf)
    state["inhibited_until"].fill(0.0)
    state["last_input_times"].fill(0.0)
    state["current_ms"] = None
    state["batch_sum"].fill(0.0)
    state["spikes"].clear()




# Обработка накопленных за текущую миллисекунду событий
def _event_processing(
        state,          # словарь параметров сети
        t_ms,           # текущая миллисекунда
        train=True,     # если True, веса меняются; иначе зафиксированы
):
    # Вектор потенциалов нейронов
    u = state["membrane_potentials"]
    # Экспоненциальное затухание потенциалов
    dt = t_ms - state["last_update"]
    u *= np.exp(-dt / h_conf.TAU_LEAK)
    state["last_update"][:] = t_ms

    # Добавляем вклад только тем нейронам, которые не подавлены
    mask_active = (
        (t_ms >= state["inhibited_until"]) &
        (t_ms >= state["last_spike"] + h_conf.T_REF)
    )
    u[mask_active] += state["batch_sum"][mask_active]

    # Если были спайки у одного или нескольких нейронов
    gave_spike = np.where(u > h_conf.I_THRES)[0]
    if gave_spike.size > 0:
        # Победителем считаем нейрон с максимальным потенциалом
        winner_index = gave_spike[np.argmax(u[gave_spike])]
        # Фиксируем спайк
        state["spikes"].append((t_ms, winner_index))
        state["last_spike"][winner_index] = t_ms
        u[winner_index] = 0.0
        # Латеральное торможение
        apply_inhibition(
            inhibited_until_array=state["inhibited_until"],
            t=t_ms,
            spiking_index=winner_index
        )
        # Если сеть обучается, то обновляем веса для победителя по правилу STDP
        if train:
            update_weights_stdp(
                neuron_index=winner_index,
                t_post=t_ms,
                weights=state["weights"],
                last_input_times=state["last_input_times"]
            )
            all_neurons = np.arange(h_conf.COUNT_NEURONS)
            losers = all_neurons[all_neurons != winner_index]
            for loser in losers:
                w = state["weights"][loser]
                dw_ltd = (0.1 * h_conf.ALPHA_MINUS) * np.exp(
                    -h_conf.BETA_MINUS * (h_conf.W_MAX - w) / h_conf.W_RANGE
                )
                w -= dw_ltd
                np.clip(w, h_conf.W_MIN, h_conf.W_MAX, out=w)

    # Очищаем накопленное
    state["batch_sum"].fill(0.0)



# Получение и накопление событий
def hidden_layer_step(
        state,          # словарь параметров сети
        event,          # событие из input_layer
        train=True      # если True, веса меняются; иначе зафиксированы
):
    # Извлекаем время события, координаты пикселя и полярность
    t, x, y, p = event
    # События группируются по миллисекундам
    t_ms = int(t)
    # Определяем индекс входа
    input_id = 2 * (y * h_conf.IMAGE_WIDTH + x) + p

    # При смене миллисекунды обрабатываем группу предыдущих событий
    if state["current_ms"] is None:
        state["current_ms"] = t_ms
    elif t_ms != state["current_ms"]:
        _event_processing(state, state["current_ms"], train)
        state["current_ms"] = t_ms

    # Накапливаем вклад от этого события
    state["batch_sum"] += state["weights"][:, input_id]
    state["last_input_times"][input_id] = t



# Обработка последней группы событий
def finalize_hidden_layer(state, train=True):
    if state["current_ms"] is not None:
        _event_processing(state, state["current_ms"], train)

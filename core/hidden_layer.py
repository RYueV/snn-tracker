import numpy as np
import core.global_config as cfg
from core.learning import update_weights_stdp


"""

Скрытый слой: события -> распределение спайков по направлениям.
Каждый нейрон связан со всеми входами.

"""


# Инициализация скрытого слоя
def init_hidden_layer():
    # Количество входов (на каждый пиксель 2 состояния)
    input_size = cfg.IMAGE_HEIGHT * cfg.IMAGE_WIDTH * 2
    # Инициализация весов
    weights = np.clip(
        np.random.normal(
            cfg.W_INIT_MEAN, cfg.W_INIT_STD, (cfg.COUNT_NEURONS, input_size)
        ),
        cfg.W_MIN, cfg.W_MAX
    ).astype(np.float32)

    return {
        # Текущее значение потенциала для каждого нейрона сети
        "u": np.zeros(cfg.COUNT_NEURONS, np.float32),
        # Время последней активации каждого входа
        "last_input_times": np.zeros(input_size, np.float32),
        # Время последнего обновления потенциала нейронов
        "last_update": 0.0,
        # Время последнего спайка каждого нейрона сети
        "last_spike": np.full(cfg.COUNT_NEURONS, -np.inf, np.float32),
        # Время окончания периода ингибирования для каждого нейрона
        "inhibited_until": np.zeros(cfg.COUNT_NEURONS, np.float32),
        # Матрица весов
        "weights": weights,
        # Массив спайков: (момент времени, номер нейрона)
        "spikes": [],
        # Вектор индивидуальных порогов нейронов
        "thresh": np.full(cfg.COUNT_NEURONS, cfg.I_THRES, np.float32)
    }




# Сброс настроек скрытого слоя
def reset_hidden_layer(state):
    state["u"].fill(0.0)
    state["last_update"] = 0.0
    state["last_spike"].fill(-np.inf)
    state["inhibited_until"].fill(0.0)
    state["last_input_times"].fill(0.0)
    state["spikes"].clear()



# Получение и накопление событий
def hidden_layer_step(
        state,          # словарь параметров сети
        event,          # событие из input_layer
        train=True      # если True, веса меняются; иначе зафиксированы
):
    # Извлекаем время события, координаты пикселя и полярность
    t, x, y, p = event
    # Определяем индекс входа
    input_id = 2 * (y * cfg.IMAGE_WIDTH + x) + p

    # Экспоненциальное затухание потенциалов нейронов
    dt = t - state["last_update"]
    if dt > 0:
        state["u"] *= np.exp(-dt / cfg.TAU_LEAK)
        # Фиксируем время последнего обновления потенциалов
        state["last_update"] = t
    
    # Добавляем вклад только тем нейронам, которые не подавлены
    mask_active = (
        (t >= state["inhibited_until"]) &
        (t >= state["last_spike"] + cfg.T_REF)
    )
    state["u"][mask_active] += state["weights"][mask_active, input_id]
    
    # Фиксируем время последней активации входа input_id
    state["last_input_times"][input_id] = t

    # Если были спайки у одного или нескольких нейронов
    gave_spike = np.where(state["u"] > state["thresh"])[0]
    if gave_spike.size > 0:
        # Создаем копию потенциалов, чтобы не изменить значения
        u_copy = state["u"].copy()

        # Генерируем небольшой случайный шум
        noise = np.random.uniform(0, 1e-3, u_copy.shape)
        # Добавляем его к реальным потенциалам, чтобы избежать случая, 
        # когда значение потенциала нескольких нейронов одинаково
        u_copy += noise
        # Победителем считаем нейрон с максимальным потенциалом
        winner_index = np.argmax(u_copy)

        # Фиксируем время спайка и номер сработавшего нейрона
        state["spikes"].append((t, winner_index))
        # Обновляем время последнего спайка победившего нейрона
        state["last_spike"][winner_index] = t
        # Сбрасываем потенциал
        state["u"][winner_index] = 0.0

        # Латеральное торможение
        mask_inhibit = np.arange(cfg.COUNT_NEURONS) != winner_index
        state["inhibited_until"][mask_inhibit] = t + cfg.T_INHIBIT

        # Если сеть обучается, то обновляем веса для победителя по правилу STDP
        if train:
            update_weights_stdp(
                t_post=t,
                synapse_weights=state["weights"][winner_index],
                last_input_times=state["last_input_times"]
            )



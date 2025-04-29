import numpy as np
import core.global_config as cfg

"""

Выходной слой: распределение спайков по направлениям -> решение о направлении сдвига

"""


# Инициализация состояния выходного слоя
def init_output_layer():
    # Количество нейронов в скрытом слое (=кол-во входов в выходном)
    count_hidden_neurons = cfg.COUNT_NEURONS
    # Количество нейронов в скрытом слое
    count_output_neurons = cfg.OUT_NEURONS

    # Матрица весов 
    weights = np.clip(
        np.random.normal(
            10.0, 2.0, (count_output_neurons, count_hidden_neurons)
        ),
        cfg.W_MIN, cfg.W_MAX
    ).astype(np.float32)

    return {
        # Вектор потенциалов нейронов выходного слоя
        "u": np.zeros(count_output_neurons, np.float32),
        # Времена пресинаптических спайков
        "last_pre": np.full(count_hidden_neurons, -np.inf, np.float32),
        # Времена постсинаптических спайков
        "last_post": np.full(count_output_neurons, -np.inf, np.float32),
        # Буфер обучаемости связей
        "eligibility": np.zeros((count_output_neurons, count_hidden_neurons), np.float32),
        # Матрица весов связей (строки - выходные нейроны, столбцы - нейроны скрытого слоя)
        "weights": weights
    }



# Сброс состояния выходного слоя
def reset_output_layer(state):
    state["u"].fill(0.0)
    state["last_pre"].fill(-np.inf)
    state["last_post"].fill(-np.inf)
    state["eligibility"].fill(0.0)
    state["weights"] = np.clip(
        np.random.normal(
            10.0, 2.0, state["weights"].shape
        ),
        cfg.W_MIN, cfg.W_MAX
    ).astype(np.float32)



# Обработка прихода пресинаптического спайка из скрытого слоя
def output_pre_spike(
        state,              
        pre_idx,            # индекс пресинаптического нейрона
        t                   # время спайка (мс)
):
    # Проходим по всем нейронам выходного слоя
    for neuron_idx in range(cfg.OUT_NEURONS):
        # Запоминаем, что нейрон скрытого слоя pre_idx мог оказать влияние
        # на активацию текущего нейрона выходного слоя neuron_idx
        state["eligibility"][neuron_idx, pre_idx] += cfg.ALPHA_PLUS
    # Обновляем время последней активации пресинаптического нейрона скрытого слоя
    state["last_pre"][pre_idx] = t



# Обработка спайка выходного нейрона
def output_post_spike(
        state, 
        post_idx,           # индекс постсинаптического нейрона
        t                   # время спайка (мс)
):
    # Вычисляем разницу времени между спайком постсинаптического нейрона (нейрон выходного слоя)
    # и спайками пресинаптических нейронов (нейронов скрытого слоя)
    dt = t - state["last_pre"]
    # Вычисляем вектор коэффициентов усиления связей:
    # если пресинаптический спайк был давно, то gain_ratio -> 0
    # иначе gain_ratio -> 1
    gain_ratio = np.exp(-dt / cfg.OUT_T_ELIG)
    state["eligibility"][post_idx, :] += cfg.ALPHA_MINUS * gain_ratio
    # Обновляем время последней активации постсинаптического нейрона выходного слоя
    state["last_post"][post_idx] = t

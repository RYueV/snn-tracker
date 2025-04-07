import numpy as np


"""Глобальные константы для STDP"""
# Инкремент веса связи
ALPHA_PLUS = 19.8231
# Декремент веса связи
ALPHA_MINUS = 17.7124
# Инкремент усиления веса связи
BETA_PLUS = 0.06644
# Декремент ослабления веса связи
BETA_MINUS = 0.22925
# Окно обучения (мс)
T_LTP = 7.9024  


"""Ограничения на веса связей"""
W_MIN = 20.0
W_MAX = 600.0
W_RANGE = W_MAX - W_MIN


"""Глобальная константа для латерального торможения (мс)"""
T_INHIBIT = 2.0




# Обновление весов нейрона по правилу STDP
def update_weights_stdp(
        neuron_index,       # идекс нейрона       
        t_post,             # время спайка нейрона
        weights,            # вектор весов нейрона
        last_input_times    # вектор, который хранит время активации каждого входа
):
    # Обходим все входы: input_id - номер входа, t_pre - время активации входа input_id
    for input_id, t_pre in enumerate(last_input_times):
        delta_t = t_post - t_pre
        w = weights[neuron_index, input_id]

        # Если входной сигнал пришел незадолго до того, как нейрон активировался
        if 0 < delta_t < T_LTP:
            # Усиливаем связь
            delta_w = ALPHA_PLUS * np.exp(-BETA_PLUS * (w - W_MIN) / W_RANGE)
            w += delta_w
        else:
            # Иначе ослабляем
            delta_w = ALPHA_MINUS * np.exp(-BETA_MINUS * (W_MAX - w) / W_RANGE)
            w -= delta_w

        # Ограничиваем вес
        weights[neuron_index, input_id] = np.clip(w, W_MIN, W_MAX)

    # Возвращаем обновленный вектор весов
    return weights[neuron_index]



# Ингибирование всех нейронов, кроме сработавшего
def apply_inhibition(inhibited_until_array, t, spiking_index):
    inhibited_until_array[:] = t + T_INHIBIT
    inhibited_until_array[spiking_index] = 0.0



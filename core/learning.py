import numpy as np

# Глобальные константы для скрытого слоя
import core.hidden_config as h_conf



# Обновление весов нейрона по правилу STDP
def update_weights_stdp(
        neuron_index,       # индекс нейрона       
        t_post,             # время спайка нейрона
        weights,            # вектор весов нейрона
        last_input_times    # вектор, который хранит время активации каждого входа
):
    # Строка весов связей нейрона со всеми возможными входами
    synapse_weights = weights[neuron_index]
    # Разница между моментами времени, когда был спайк и когда пришел входной сигнал
    delta_t = t_post - last_input_times
    # Окно времени, в пределах которого можно считать, что входной сигнал пришел незадолго до спайка
    ltp_mask = np.logical_and(delta_t > 0.0, delta_t < h_conf.T_LTP)

    # Вектор коэффициентов усиления связей
    dw_ltp = h_conf.ALPHA_PLUS * np.exp(-h_conf.BETA_PLUS * (synapse_weights - h_conf.W_MIN) / h_conf.W_RANGE)
    # Вектор коэффициентов ослабления связей
    dw_ltd = h_conf.ALPHA_MINUS * np.exp(-h_conf.BETA_MINUS * (h_conf.W_MAX - synapse_weights) / h_conf.W_RANGE)

    # Если входной сигнал пришел незадолго до того, как нейрон активировался, усиливаем связь
    synapse_weights +=  ltp_mask * dw_ltp 
    # Иначе ослабляем
    synapse_weights -= (~ltp_mask) * dw_ltd
    # Ограничиваем веса в допустимом диапазоне
    np.clip(synapse_weights, h_conf.W_MIN, h_conf.W_MAX, out=synapse_weights)



# Ингибирование всех нейронов, кроме сработавшего
def apply_inhibition(inhibited_until_array, t, spiking_index):
    inhibited_until_array[:] = t + h_conf.T_INHIBIT
    inhibited_until_array[spiking_index] = 0.0


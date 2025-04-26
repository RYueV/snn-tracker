import numpy as np

# Глобальные константы для скрытого слоя
import core.hidden_config as h_conf
# Глобальные константы для выходного слоя
import core.output_config as o_conf
_exp = np.exp



# Обработка входного сигнала для LIF нейрона скрытого слоя
def process_lif_hidden(
        input_time,         # время входного события
        u,                  # текущий мембранный потенциал нейрона
        last_update,        # время последнего обновления потенциала
        last_spike,         # время последнего спайка
        inhibited_until,    # время, до которого действует ингибирование
        weight              # вес связи нейрона со входом
):
    """

    Возвращает True, если был спайк.

    """
    # Экспоненциальное затухание потенциала
    delta_t = input_time - last_update
    u = u * _exp(-delta_t / h_conf.TAU_LEAK)

    # Если не вышло время ингибирования и не закончился рефрактерный период
    if (input_time < inhibited_until) or (input_time < last_spike + h_conf.T_REF):
        # Игнорируем входной сигнал
        return False, u

    # Иначе увеличиваем на вес связи нейрона со входом
    u += weight
    # Если потенциал нейрона больше порогового -> спайк, u = 0
    if u > h_conf.I_THRES:
        return True, 0.0
    
    return False, u




# Обработка входного сигнала для LIF нейрона выходного слоя
def process_lif_output(
        input_time,         # время входного события
        u,                  # текущий мембранный потенциал нейрона
        last_update,        # время последнего обновления потенциала
        last_spike,         # время последнего спайка
        inhibited_until,    # время, до которого действует ингибирование
        weight              # вес связи нейрона со входом
):
    """

    Возвращает True, если был спайк.

    """
    # Экспоненциальное затухание потенциала
    delta_t = input_time - last_update
    u = u * _exp(-delta_t / o_conf.TAU_LEAK)

    # Если не вышло время ингибирования и не закончился рефрактерный период
    if (input_time < inhibited_until) or (input_time < last_spike + o_conf.T_REF):
        # Игнорируем входной сигнал
        return False, u

    # Иначе увеличиваем на вес связи нейрона со входом
    u += weight
    # Если потенциал нейрона больше порогового -> спайк, u = 0
    if u > o_conf.I_THRES:
        return True, 0.0
    
    return False, u


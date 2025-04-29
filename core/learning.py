import numpy as np
import core.global_config as cfg



# Обновление весов победившего нейрона по правилу STDP
def update_weights_stdp(     
        t_post,             # время спайка нейрона
        synapse_weights,    # вектор весов нейрона
        last_input_times    # вектор, который хранит время активации каждого входа
):
    # Разница между моментами времени, когда был спайк и когда пришел входной сигнал
    delta_t = t_post - last_input_times
    # Окно времени, в пределах которого можно считать, что входной сигнал пришел незадолго до спайка
    ltp_mask = (delta_t > 0.0) & (delta_t < cfg.T_LTP)

    # Вектор коэффициентов усиления связей
    dw_ltp = cfg.ALPHA_PLUS * np.exp(-cfg.BETA_PLUS * (synapse_weights - cfg.W_MIN) / cfg.W_RANGE)
    # Вектор коэффициентов ослабления связей
    dw_ltd = cfg.ALPHA_MINUS * np.exp(-cfg.BETA_MINUS * (cfg.W_MAX - synapse_weights) / cfg.W_RANGE)

    # Если входной сигнал пришел незадолго до того, как нейрон активировался, усиливаем связь
    synapse_weights +=  ltp_mask * dw_ltp 
    # Иначе ослабляем
    synapse_weights -= (~ltp_mask) * dw_ltd
    # Ограничиваем веса в допустимом диапазоне
    np.clip(synapse_weights, cfg.W_MIN, cfg.W_MAX, out=synapse_weights)




# p-STDP для выходного слоя
def apply_reward_pstdp(
        weights,                # матрица весов выходного слоя
        eligibility,            # буфер обучаемости
        reward                  # награда (+1, -1)
):
    """
    
    В буфере eligibility содержатся значения весов связей, накопленные при обработке
    постсинтаптических и пресинаптических спайков;
    Эти значения обратно пропорциональны разнице времени между спайками нейронов
    скрытого и выходного слоев: если разница мала, то значения весов (в буфере) больше и наоборот;
    Если награда положительная, то есть связь, зафиксированная в eligibility, подтверждается,
    то веса связей усиливаются уже в настоящей матрице весов, если отрицательная - уменьшаются.
    
    """
    weights += cfg.OUT_ETA * reward * eligibility
    np.clip(weights, cfg.W_MIN, cfg.W_MAX, out=weights)




# Экспоненциальное затухание устаревших значений в eligibility
def decay_eligibility(
        eligibility,            # буфер обучаемости
        dt                      # шаг времени (мс)
):
    eligibility *= np.exp(-dt / cfg.OUT_T_ELIG)
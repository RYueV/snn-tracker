"""Настройки симуляции"""
# Размер окна обзора камеры (в пикселях)
IMAGE_HEIGHT = 28
IMAGE_WIDTH  = 28


"""Константы скрытого слоя"""
COUNT_NEURONS = 16        # количество нейронов в скрытом слое

# Параметры LIF-нейронов
TAU_LEAK   = 25.0         # постоянная времени утечки (мс)
I_THRES    = 120.0        # порог спайка
T_REF      = 8.0          # рефрактерный период (мс)
T_INHIBIT  = 4.0          # период ингибирования (мс)

# Константы STDP
ALPHA_PLUS  = 2.0         # инкремент веса связи
ALPHA_MINUS = 2.0         # декремент веса связи
BETA_PLUS   = 0.6         # инкремент усиления веса связи
BETA_MINUS  = 0.6         # декремент ослабления веса связи
T_LTP       = 15.0        # окно обучения (мс)

# Начальные веса
W_INIT_MEAN = 40.0              # среднее значение веса 
W_INIT_STD  = 10.0              # дисперсия
W_MIN       = 20.0              # минимальное значение
W_MAX       = 400.0             # максимальное значение
W_RANGE     = W_MAX - W_MIN     # диапазон значений


"""Константы выходного слоя"""
OUT_NEURONS  = 8            # количество нейронов в выходном слое

# Параметры LIF-нейронов
OUT_TAU_LEAK   = 50.0       # постоянная времени утечки (мс)
OUT_I_THRES    = 200.0      # порог спайка
OUT_T_REF      = 10.0       # рефрактерный период (мс)

# Константы p-STDP
OUT_T_ELIG = 150.0          # время жизни значений в буфере обучаемости связей eligibility (мс)
OUT_ETA    = 2e-3           # скорость обучения

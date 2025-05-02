import numpy as np


"""

Входной слой: кадры с камеры -> события

"""


# Рефрактерный период для одинаковых событий (мс)
PIXEL_REF  = 8.0
# Логарифмический порог изменения яркости для генерации события
LOG_THRESHOLD = 0.2
# Константы
EXP_TH = np.exp(LOG_THRESHOLD)
EXP_MTH = np.exp(-LOG_THRESHOLD)
EPS = 1e-5




# Инициализация состояния генератора событий
def init_event_generator(
        frame_shape=(28, 28)        # размер изображения (обзор камеры)
):
    return {
        # Для каждого пикселя хранит яркость, при которой было последнее on-событие
        "last_on": np.ones(frame_shape, dtype=np.float32) * 0.01,
        # Для каждого пикселя хранит яркость, при которой было последнее off-событие
        "last_off": np.ones(frame_shape, dtype=np.float32) * 1.0,
        # Для каждого пикселя и каждой полярности хранит время последнего события
        "t_last_event": np.full((*frame_shape, 2), -np.inf, dtype=np.float32)
    }




# Генерация событий между двумя кадрами
def generate_events(
        state,              # состояние генератора (словарь)
        old_frame,          # предыдущий кадр (2D np матрица)
        new_frame,          # новый кадр (2D np матрица)
        prev_t,             # время фиксации предыдущего кадра
        new_t               # время фиксации нового кадра
):
    """

    Событие - кортеж (t, x, y, p):
        t - время, когда пиксель изменил яркость сильнее, чем в exp(threshold) раз
        x, y - координаты пикселя
        p - полярность (p=1, если пиксель стал ярче; p=0, если темнее)
        p=1 => on-событие, p=0 => off-событие

    """
    # Размер кадра
    height, width = old_frame.shape
    # Интервал между кадрами в мс (>0)
    dt = new_t - prev_t + EPS
    # Список событий
    events = []

    # Проходим по всем пикселям
    for y in range(height):
        for x in range(width):
            # Берем значение яркости с предыдущего и текущего кадров
            old_val = old_frame[y, x]
            new_val = new_frame[y, x]

            # Пропускаем черные пиксели (важно только перемещение объекта)
            if old_val == 0.0 and new_val == 0.0:
                continue

            # Изменение яркости между кадрами
            dI = new_val - old_val
            # Производная яркости по времени
            dI_dt = dI / dt

            # Генерация on-событий
            # На сколько порядков увеличилась яркость с момента фиксации последнего on-события
            log_ratio = np.log((new_val + EPS) / (state["last_on"][y, x] + EPS))
            # Если изменение яркости перешло порог и яркость увеличивается
            if log_ratio >= LOG_THRESHOLD and dI_dt > 0:
                # Рассчитываем новый порог яркости для фиксации следующего on-события
                target = state["last_on"][y, x] * EXP_TH
                # Вычисляем момент времени пересечения порога с помощью линейной интерполяции
                t_cross = prev_t + (target - old_val) / (dI_dt + EPS)
                # Добавляем случайное смещение времени
                t_cross += np.random.uniform(0, 5)
                # prev_t <= t_cross <= new_t
                t_cross = np.clip(t_cross, prev_t, new_t)
                # Событие фиксируется только если рефрактерный период вышел
                if t_cross - state["t_last_event"][y, x, 1] >= PIXEL_REF:
                    events.append((t_cross, x, y, 1))
                    # Обновляем порог яркости для генерации следующего on-события
                    state["last_on"][y, x] = target
                    # Обновляем время фиксации последнего on-события
                    state["t_last_event"][y, x, 1] = t_cross

            # Генерация off-событий
            # На сколько порядков уменьшилась яркость с момента фиксации последнего off-события
            log_ratio = np.log((new_val + EPS) / (state["last_off"][y, x] + EPS))
            # Если изменение яркости перешло порог и яркость уменьшается
            if log_ratio <= -LOG_THRESHOLD and dI_dt < 0:
                # Рассчитываем новый порог яркости для фиксации следующего off-события
                target = state["last_off"][y, x] * EXP_MTH
                # Вычисляем момент времени пересечения порога с помощью линейной интерполяции
                t_cross = prev_t + (target - old_val) / (dI_dt + EPS)
                # Добавляем случайное смещение времени
                t_cross += np.random.uniform(0, 5)
                # prev_t <= t_cross <= new_t
                t_cross = np.clip(t_cross, prev_t, new_t)
                # Событие фиксируется только если рефрактерный период вышел
                if t_cross - state["t_last_event"][y, x, 0] >= PIXEL_REF:
                    events.append((t_cross, x, y, 0))
                    # Обновляем порог яркости для генерации следующего off-события
                    state["last_off"][y, x] = target
                    # Обновляем время фиксации последнего off-события
                    state["t_last_event"][y, x, 0] = t_cross

    # Сортируем события по времени
    events.sort(key=lambda x: x[0])
    return events

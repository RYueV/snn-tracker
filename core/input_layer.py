import numpy as np

"""

Входной слой: кадры с камеры -> события

"""


# Рефрактерный период для одинаковых событий (мс)
PIXEL_REF = 5.0
# Логарифмический порог изменения яркости для генерации события
THRESHOLD = 0.2
EXP_THRESHOLD = np.exp(THRESHOLD)
EXP_MINUS_THRESHOLD = np.exp(-THRESHOLD)
# Малая константа для корректных вычислений
EPS = 1e-5




# Инициализация состояния генератора событий
def init_event_generator(
        frame_shape=(28, 28)        # размер изображения (обзор камеры)
):
    return {
        # Для каждого пикселя хранит яркость, при которой было последнее on-событие
        "last_pixel_value_on": np.ones(frame_shape, dtype=np.float32) * 0.01,
        # Для каждого пикселя хранит яркость, при которой было последнее off-событие
        "last_pixel_value_off": np.ones(frame_shape, dtype=np.float32) * 1.0,
        # Для каждого пикселя и каждой полярности хранит время последнего события
        "t_last_event": np.full((*frame_shape, 2), -np.inf, dtype=np.float32),
        # Случайный сдвиг времени для каждого пикселя
        "pixel_time_offset": np.random.uniform(-2, 2, size=frame_shape).astype(np.float32)
    }




# Бинарный поиск времени пересечения порога яркости
def _find_threshold_time_binary_search(
        old_value,          # яркость в момент времени t0
        new_value,          # яркость в момент времени t1
        target_value,       # порог яркости
        t0, t1,             # известные точки, t0 < t < t1
        max_iter=10         # количество итераций поиска
):
    delta_old = old_value - target_value
    delta_new = new_value - target_value

    # Если разности имеют одинаковый знак, то на интервале [t0, t1] яркость не пересекает порог
    if delta_old * delta_new > 0:
        return None

    left, right = t0, t1
    # Вычисление констант для линейной интерполяции яркости
    denom = max(t1 - t0, EPS)
    inv_denom = 1.0 / denom
    diff_val = new_value - old_value

    for _ in range(max_iter):
        # Делим интервал пополам
        middle = 0.5 * (left + right)
        # Ищем яркость в момент времени middle с помощью линейной интерполяции
        val_mid = old_value + diff_val * ((middle - t0) * inv_denom)
        delta_mid = val_mid - target_value
        # Если нашли порог, возвращаем время
        if delta_mid == 0:
            return middle
        # Выбираем половину с корнем
        if delta_mid * delta_old > 0:
            left = middle
            delta_old = delta_mid
        else:
            right = middle

    return 0.5 * (left + right)




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
    # Кэширование функций
    _log = np.log
    _clip = np.clip

    # Размер изображения
    height, width = old_frame.shape
    # Список событий
    events = []

    # Последние яркости при фиксации on-событий
    last_on_value_arr = state["last_pixel_value_on"]
    # Последние яркости при фиксации off-событий
    last_off_value_arr = state["last_pixel_value_off"]
    # Моменты времени фиксации последних событий
    t_last_event_arr = state["t_last_event"]
    # Случайный сдвиг времени для каждого пикселя
    time_offset_arr = state["pixel_time_offset"]


    # Обходим все пиксели
    for y in range(height):
        for x in range(width):
            old_value = old_frame[y, x]
            new_value = new_frame[y, x]

            # Пропускаем полностью черные участки
            if new_value == 0.0 and old_value == 0.0:
                continue

            # Проверяем наличие on-событий
            while True:
                # Берем значение яркости, при которой было последнее on-событие
                last_on = last_on_value_arr[y, x] + EPS
                # Если с того момента яркость увеличилась менее чем в e^(threshold) раз, пропускаем
                if _log((new_value + EPS) / last_on) < THRESHOLD:
                    break
                # Вычисляем порог яркости для генерации on-события
                target_on = last_on * EXP_THRESHOLD
                # Ищем точный момент времени, когда должны были достичь порога
                t_cross = _find_threshold_time_binary_search(
                    old_value=old_value,
                    new_value=new_value,
                    target_value=target_on,
                    t0=prev_t,
                    t1=new_t
                )
                # Если не смогли определить, пропускаем
                if t_cross is None:
                    break
                # Добавляем случайный сдвиг и обрезаем по времени
                t_cross = _clip(t_cross + time_offset_arr[y, x], prev_t, new_t)
                # Проверяем рефрактерный период
                if t_cross - t_last_event_arr[y, x, 1] < PIXEL_REF:
                    break
                # Фиксируем on-событие
                events.append((t_cross, x, y, 1))
                last_on_value_arr[y, x] = target_on
                t_last_event_arr[y, x, 1] = t_cross

            # Проверяем наличие off-событий
            while True:
                # Берем значение яркости, при которой было последнее off-событие
                last_off = last_off_value_arr[y, x] + EPS
                # Если с того момента яркость уменьшилась менее чем в e^(threshold) раз, пропускаем
                if _log((new_value + EPS) / last_off) > -THRESHOLD:
                    break
                # Вычисляем порог яркости для генерации off-события
                target_off = last_off * EXP_MINUS_THRESHOLD
                # Ищем точный момент времени, когда должны были достичь порога
                t_cross = _find_threshold_time_binary_search(
                    old_value=old_value,
                    new_value=new_value,
                    target_value=target_off,
                    t0=prev_t,
                    t1=new_t
                )
                # Если не смогли определить, пропускаем
                if t_cross is None:
                    break
                # Добавляем случайный сдвиг и обрезаем по времени
                t_cross = _clip(t_cross + time_offset_arr[y, x], prev_t, new_t)
                # Проверяем рефрактерный период
                if t_cross - t_last_event_arr[y, x, 0] < PIXEL_REF:
                    break
                # Фиксируем off-событие
                events.append((t_cross, x, y, 0))
                last_off_value_arr[y, x] = target_off
                t_last_event_arr[y, x, 0] = t_cross

    # Сортируем события по времени
    events.sort(key=lambda x: x[0])
    return events

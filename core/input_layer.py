import numpy as np

"""

Входной слой: кадры с камеры -> события

"""


# Инициализация состояния генератора событий
def init_event_generator(
        frame_shape=(28,28),        # размер изображения (обзор камеры)
        pixel_ref=3.0               # рефрактерный период для пикселя
):
    return {
        # Для каждого пикселя хранит яркость, при которой было последнее on-событие
        "last_pixel_value_on": np.ones(frame_shape, dtype=np.float32) * 0.01,
        # Для каждого пикселя хранит яркость, при которой было последнее off-событие
        "last_pixel_value_off": np.ones(frame_shape, dtype=np.float32) * 1.0,
        # Для каждого пикселя и каждой полярности хранит время последнего события
        "t_last_event": np.full((*frame_shape, 2), -np.inf, dtype=np.float32),
        # Рефрактерный период для пикселя с одной и той же полярностью
        "pixel_ref": pixel_ref
    }



# Генерация событий между двумя кадрами
def generate_events(
        state,              # состояние генератора (словарь)
        old_frame,          # предыдущий кадр (2D np матрица)
        new_frame,          # новый кадр (2D np матрица)
        prev_t,             # время фиксации предыдущего кадра
        new_t,              # время фиксации нового кадра
        threshold=0.3       # логарифмический порог изменения яркости для генерации события
):
    """

    Событие - кортеж (t, x, y, p):
        t - время, когда пиксель изменил яркость сильнее, чем в exp(threshold) раз
        x, y - координаты пикселя
        p - полярность (p=1, если пиксель стал ярче; p=0, если темнее)
        p=1 => on-событие, p=0 => off-событие

    Генератор событий хранит матрицы опорных яркостей:
        1) last_pixel_value_on для каждого пикселя хранит яркость, при которой было последнее on-событие
        2) last_pixel_value_off для каждого пикселя хранит яркость, при которой было последнее off-событие

    Событие возникает, если текущая яркость пикселя меньше/больше опорной в exp(threshold) раз.
    Сравнение с опорной яркостью, а не с предыдущим кадром повышает устойчивость к шумам.

    Для того, чтобы все события не происходили в момент времени t, используется линейная интерполяция
    (предполагается, что яркость пикселя линейно меняется от значения на old_frame до значения на new_frame).

    """
    # Размер изображения
    height, width = old_frame.shape
    # Список событий
    events = []

    # Матрицы состояния генератора
    last_pixel_value_on = state["last_pixel_value_on"]      # последние яркости при фиксации on-событий
    last_pixel_value_off = state["last_pixel_value_off"]    # последние яркости при фиксации off-событий
    t_last_event = state["t_last_event"]                    # моменты времени фиксации последних событий
    
    # Рефрактерный период для пикселя с одной и той же полярностью
    pixel_ref = state["pixel_ref"]     
    # Малая константа для защиты от log(0)         
    eps = 1e-5  

    # Обходим все пиксели
    for y in range(height):
        for x in range(width):
            # Яркость пикселя на прошлом кадре
            old_pixel_value = old_frame[y, x]
            # Яркость пикселя на новом кадре
            new_pixel_value = new_frame[y, x]

            # Если яркость изменилась незначительно, сразу пропускаем
            delta_pixels = new_pixel_value - old_pixel_value
            if abs(delta_pixels) < 1e-12:
                continue

            # Проверяем наличие on-событий
            while True:
                # Берем значение яркости, при которой было последнее on-событие
                last_recorded_value = last_pixel_value_on[y, x] + eps

                # Если с того момента яркость изменилась менее чем в e^(threshold) раз
                if np.log((new_pixel_value + eps) / last_recorded_value) < threshold:
                    # Выходим из цикла, тк изменение незначительное
                    break

                # Фиксируем яркость, при которой достигли нового on-события
                new_recorded_value = last_recorded_value * np.exp(threshold)
                # Используем линейную интерполяцию, чтобы найти момент времени события
                alpha = (new_recorded_value - old_pixel_value) / delta_pixels
                t = prev_t + alpha * (new_t - prev_t)

                # Если вышли за временные рамки, или рефрактерный период не вышел, прерываем цикл
                if not (prev_t <= t <= new_t):
                    break
                if t - t_last_event[y, x, 1] < pixel_ref:
                    break

                # Фиксируем on-событие
                events.append((t, x, y, 1)) 
                # Обновляем значение яркости, при которой было последнее on-событие
                last_pixel_value_on[y, x] = new_recorded_value
                # Обновляем время, в которое было последнее on-событие
                t_last_event[y, x, 1] = t


            # Проверяем наличие off-событий
            while True:
                # Берем значение яркости, при которой было последнее off-событие
                last_recorded_value = last_pixel_value_off[y, x] + eps

                # Если с того момента яркость изменилась менее чем в e^(-threshold) раз
                if np.log((new_pixel_value + eps) / last_recorded_value) > -threshold:
                    # Выходим из цикла, тк изменение незначительное
                    break

                # Фиксируем яркость, при которой достигли нового off-события
                new_recorded_value = last_recorded_value * np.exp(-threshold)
                # Используем линейную интерполяцию, чтобы найти момент времени события
                alpha = (new_recorded_value - old_pixel_value) / delta_pixels
                t = prev_t + alpha * (new_t - prev_t)

                # Если вышли за временные рамки, или рефрактерный период не вышел, прерываем цикл
                if not (prev_t <= t <= new_t):
                    break
                if t - t_last_event[y, x, 0] < pixel_ref:
                    break
                
                # Фиксируем off-событие
                events.append((t, x, y, 0))  
                # Обновляем значение яркости, при которой было последнее off-событие
                last_pixel_value_off[y, x] = new_recorded_value
                # Обновляем время, в которое было последнее off-событие
                t_last_event[y, x, 0] = t

    return events

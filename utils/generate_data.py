import numpy as np
import random
import hashlib
from .data_converter import save_pickle



# Виды траекторий движения объекта
TRAJECTORY_STYLES = [
    "linear",   # линейное движение строго по направлению
    "noisy",    # добавление случайного шума к основному направлению каждый кадр
    "curved",   # постоянное отклонение от основного направления вбок
    "impulse"   # добавление резкого скачка в середине пути
]
# Размер кадра (ширина, высота) в пикселях
FIELD_SIZE = (28, 28)
# Сторона квадрата в пикселях
SQUARE_SIZE = 7
# Половина стороны квадрата
HALF_SIZE = SQUARE_SIZE // 2
# Количество кадров на один пример
FRAMES_PER_SAMPLE = 9
# Базовые направления движения объекта (dx, dy)
DIRECTIONS = [
    (0, 1),         # вниз
    (1, 0),         # вправо
    (1, 1),         # вправо-вниз
    (0, -1),        # вверх
    (-1, 0),        # влево
    (-1, -1),       # влево-вверх
    (1, -1),        # вправо-вверх
    (-1, 1)         # влево-вниз   
]



# Генерация шумов по осям в зависимости от типа траектории
def generate_noise(
        direction,      # направление движения (dx, dy)
        max_noise,      # максимально возможное значение шума
        style           # вид траектории
):
    dx, dy = direction
    noise_dxdy = []

    # Шум генерируем для каждого кадра внутри одного примера
    for frame_i in range(FRAMES_PER_SAMPLE):
        # Добавление случайного шума к основному направлению каждый кадр
        if style == "noisy":
            noise_dx = np.random.randint(-max_noise, max_noise + 1)
            noise_dy = np.random.randint(-max_noise, max_noise + 1)
        # Постоянное отклонение от основного направления вбок
        elif style == "curved":
            if dx != 0:
                noise_dx = 0
                noise_dy = random.choice([-max_noise, max_noise])
            else:
                noise_dx = random.choice([-max_noise, max_noise])
                noise_dy = 0
        # Добавление резкого скачка в середине пути
        elif style == "impulse" and frame_i == FRAMES_PER_SAMPLE // 2:
            noise_dx = random.choice([-3, 3])
            noise_dy = random.choice([-3, 3])
        # Если linear, то шума нет
        else:
            noise_dx = noise_dy = 0
        noise_dxdy.append((noise_dx, noise_dy))
    
    return noise_dxdy



# Проверка выхода объекта за границы
def check_trajectory(
        direction,      # направление движения
        start_pos,      # стартовая координата
        speed,          # на сколько пикселей сдвигается объект за кадр
        noise_dxdy      # значение шума по x и по y: список кортежей (noise_dx, noise_dy)
):
    height, width = FIELD_SIZE
    dx, dy = direction
    cx, cy = start_pos

    # Проверяем каждую точку траектории
    for noise_dx, noise_dy in noise_dxdy:
        cx += dx * speed + noise_dx
        cy += dy * speed + noise_dy

        # Если квадрат не помещается в поле, возвращаем False
        if (cx - HALF_SIZE < 0) or (cx + HALF_SIZE >= width):
            return False
        if (cy - HALF_SIZE < 0) or (cy + HALF_SIZE >= height):
            return False

    return True



# Генерация одного примера (последовательности кадров)
def generate_one_sample(
        direction,              # (dx, dy)
        start_pos,              # (cx, cy) начальная позиция центра
        speed,                  # скорость в пикселях/кадр
        noise_dxdy,             # величина случайного отклонения
        style,
        square=None             # матрица яркостей квадрата
):
    height, width = FIELD_SIZE
    dx, dy = direction
    cx, cy = start_pos
    frames = []

    # Создаем заданное количество кадров
    for frame_i in range(FRAMES_PER_SAMPLE):
        # Создаем кадр (черное поле)
        frame = np.zeros(FIELD_SIZE, dtype=np.float32)
        # Шум на данном кадре
        noise_dx, noise_dy = noise_dxdy[frame_i]

        # Обновляем координаты центра квадрата
        cx += dx * speed + noise_dx
        cy += dy * speed + noise_dy

        # Ограничиваем координаты, чтобы квадрат помещался в поле
        cx = np.clip(cx, HALF_SIZE, width - HALF_SIZE - 1)
        cy = np.clip(cy, HALF_SIZE, height - HALF_SIZE - 1)

        # Координаты углов квадрата
        x_min = int(cx - HALF_SIZE)
        x_max = int(cx + HALF_SIZE) + 1
        y_min = int(cy - HALF_SIZE)
        y_max = int(cy + HALF_SIZE) + 1

        # Рисуем квадрат
        frame[y_min:y_max, x_min:x_max] = square
        # Сохраняем кадр
        frames.append(frame)

    return {
        "frames": frames,
        "direction": direction,
        "start_pos": start_pos,
        "speed": speed,
        "noise": noise_dxdy,
        "style": style
    }




# Запоминаем примеры для отсечения дубликатов
def _hash_frames(frames):
    frames_array = np.stack(frames, axis=0)     # (num_frames, h, w)
    frames_byte = (frames_array * 255).astype(np.uint8).tobytes()
    h = hashlib.md5(frames_byte).hexdigest()
    return h



# Генерация полного датасета
def generate_dataset(
        samples_per_style=40,   # количество примеров на один вид траектории
        speed_var=[1,2],        # возможные значения количества пикселей, на которое смещается квадрат за кадр
        noise_var=[0,1],        # возможные значения максимальной амплитуды шума
        color_var=True,         # если True, квадрат разноцветный, иначе белый
        max_tries=10000,        # количество попыток генерации примера
        save_path="data/dataset.pkl"    # путь для сохранения датасета
):
    dataset = []
    height, width = FIELD_SIZE
    min_start = HALF_SIZE
    max_start_x = width - HALF_SIZE - 1
    max_start_y = height - HALF_SIZE - 1
    seen = set()

    for _ in range(samples_per_style):
        # Генерируем квадрат
        if color_var:
            square = np.clip(
                np.random.normal(loc=1.0, scale=0.15, size=(SQUARE_SIZE, SQUARE_SIZE)),
                0.5, 1.0
            )
        else:
            square = np.ones((SQUARE_SIZE, SQUARE_SIZE))

        # Выбираем шум и скорость
        max_noise = random.choice(noise_var)
        speed = random.choice(speed_var)

        # Перебираем все виды траекторий
        for style in TRAJECTORY_STYLES:
            tries = 0
            # Будем пробовать разные стартовые позиции, пока не найдем минимум 3 допустимых направления
            while tries < max_tries:
                tries += 1
                # Генерируем начальную позицию центра квадрата
                start_pos = (
                    random.randint(min_start, max_start_x),
                    random.randint(min_start, max_start_y)
                )

                # Список успешно сгенерированных направлений
                valid_directions = []

                # Перебираем все базовые направления
                for direction in DIRECTIONS:
                    # Генерируем шум
                    noise_dxdy = generate_noise(direction, max_noise, style)
                    # Проверяем, не выйдет ли квадрат за границы поля
                    if not check_trajectory(direction, start_pos, speed, noise_dxdy):
                        continue
                    # Генерируем пример
                    sample = generate_one_sample(direction, start_pos, speed, noise_dxdy, style, square)
                    # Хэшируем
                    sample_h = _hash_frames(sample["frames"])
                    # Пропускаем дубликаты
                    if sample_h in seen:
                        continue
                    # Добавляем в датасет
                    seen.add(sample_h)
                    dataset.append(sample)
                    valid_directions.append(direction)

                # Если набрали хотя бы 3 допустимых направления, выходим из цикла выбора позиции
                if len(valid_directions) >= 3:
                    break

    save_pickle(save_path=save_path, data=dataset)
    return len(dataset)

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
FRAME_SIZE = (28, 28)
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
# Количество примеров на одно направление
SAMPLES_PER_DIR = 100



# Генерация шумов по осям в зависимости от типа траектории
def generate_noise(
        direction,      # направление движения (dx, dy)
        max_noise,      # максимально возможное значение шума
        style           # вид траектории
):
    dx, _ = direction
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
    height, width = FRAME_SIZE
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
    height, width = FRAME_SIZE
    dx, dy = direction
    cx, cy = start_pos
    frames = []

    # Создаем заданное количество кадров
    for frame_i in range(FRAMES_PER_SAMPLE):
        # Создаем кадр (черное поле)
        frame = np.zeros(FRAME_SIZE, dtype=np.float32)
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
        speed_var=[1,2],        # возможные значения количества пикселей, на которое смещается квадрат за кадр
        noise_var=[0,1],        # возможные значения максимальной амплитуды шума
        color_var=True,         # если True, квадрат разноцветный, иначе белый
        max_tries=10000,        # количество попыток генерации примера
        save_path="data/dataset.pkl"    # путь для сохранения датасета
):
    dataset = []
    # Ограничения на начальные позиции объекта
    min_start = HALF_SIZE
    max_start_x = FRAME_SIZE[1] - HALF_SIZE - 1
    max_start_y = FRAME_SIZE[0] - HALF_SIZE - 1
    # Множество тех примеров, которые уже видели, чтобы не повторяться
    seen = set()
    # Количество сгенерированных примеров на каждый стиль для каждого направления
    style_counter = {d: {s: 0 for s in TRAJECTORY_STYLES} for d in DIRECTIONS}
    # Минимальное количество примеров на пару (стиль, направление)
    min_samp_per_style = SAMPLES_PER_DIR // len(TRAJECTORY_STYLES)
    # Вспомогательный словарь, который определяет, нужны ли дополнительные примеры на пару,
    # если в результате первоначального распределения остаются "лишние" примеры
    styles_with_add_samp = {
        s: (i < SAMPLES_PER_DIR % len(TRAJECTORY_STYLES))
        for i,s in enumerate(TRAJECTORY_STYLES)
    }

    # Пока не набрали нужное количество примеров хотя бы для одного из направлений
    while any(sum(style_counter[d].values()) < SAMPLES_PER_DIR for d in DIRECTIONS):
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
                valid_directions = 0

                # Перебираем все базовые направления
                for direction in DIRECTIONS:
                    # Проверяем, не набрали ли нужное количество примеров на (стиль, направление)
                    target = min_samp_per_style + int(styles_with_add_samp[style])
                    if style_counter[direction][style] >= target:
                        continue
                    # Проверяем, не набрали ли необходимое количество примеров на направление
                    if sum(style_counter[direction].values()) >= SAMPLES_PER_DIR:
                        continue

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
                    style_counter[direction][style] += 1
                    valid_directions += 1

                    # Если примеров достаточно, можно завершать
                    if all(sum(style_counter[d].values()) >= SAMPLES_PER_DIR for d in DIRECTIONS):
                        break

                # Нужно минимум 3 направления6
                if valid_directions > 2:
                    break


    save_pickle(save_path=save_path, data=dataset)

    print(f"Сгенерировано примеров: {len(dataset)}")
    print("Распределение примеров по направлениям и видам траекторий\n")
    print(style_counter)

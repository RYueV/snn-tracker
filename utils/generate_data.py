import numpy as np
import random

from .data_converter import save_pickle


TRAJECTORY_STYLES = ["linear", "noisy", "curved", "impulse"]



# Генерация одной траектории объекта
def generate_one_sample(
        direction,              # направление движения (dx, dy)
        start_pos,              # начальная позиция центра объекта (cx, cy)
        speed,                  # скорость движения объекта в пикселях за кадр
        noise,                  # максимальное случайное отклонение объекта от основного направления
        frames_count,           # количество кадров в примере
        field_size=(28,28),     # размер изображения (высота, ширина)
        square_size=7,          # сторона квадрата (объекта)
        style="linear",         # вид тректории
        square=None             # массив нормализованных яркостей для квадрата
):
    height, width = field_size
    dx, dy = direction
    cx, cy = start_pos
    frames = []
    half = square_size // 2

    # Генерируем нужное количество кадров
    for frame_i in range(frames_count):
        # Создаем пустое изображение
        frame = np.zeros(field_size, dtype=np.float32)
        
        # Выбор смещения в зависимости от стиля движения
        if style == "noisy":
            # Случайное смещение и по x, и по y
            noise_dx = np.random.randint(-noise, noise + 1)
            noise_dy = np.random.randint(-noise, noise + 1)
        elif style == "curved":
            # При движении по горизонтали добавляем смещение по вертикали и наоборот
            if dx != 0:
                noise_dx = 0
                noise_dy = random.choice([-noise, noise])
            else:
                noise_dx = random.choice([-noise, noise])
                noise_dy = 0
        elif style == "impulse" and frame_i == frames_count // 2:
            # Резкое смещение по направлениям 
            noise_dx = random.choice([-3, 3])
            noise_dy = random.choice([-3, 3])
        else:
            noise_dx = 0
            noise_dy = 0

        # Обновляем координаты центра объекта с учетом базового перемещения и случайного смещения
        cx += dx * speed + noise_dx
        cy += dy * speed + noise_dy

        # Ограничиваем координаты так, чтобы весь квадрат оставался внутри изображения
        cx = np.clip(cx, half, width - half - 1)
        cy = np.clip(cy, half, height - half - 1)

        # Определяем границы квадрата
        x_min = int(cx - half)
        x_max = int(cx + half) + 1  # +1, чтобы включить последний пиксель
        y_min = int(cy - half)
        y_max = int(cy + half) + 1

        # Если не передан массив яркостей для квадрата, генерируем просто белый квадрат
        if square is None:
            square = np.ones((square_size, square_size), dtype=np.float32)

        # Рисуем квадрат
        frame[y_min:y_max, x_min:x_max] = square
        frames.append(frame)

    return {
        "frames": frames,
        "direction": direction,
        "start_pos": start_pos,
        "speed": speed,
        "noise": noise,
        "style": style,
        "square_size": square_size
    }



# Проверка, что при движении объекта он не выходит за пределы кадра
def check_trajectory(
        direction,          # направление движения (dx, dy)
        start_pos,          # начальная позиция центра объекта (cx, cy)
        speed,              # скорость движения объекта в пикселях за кадр
        noise,              # максимальное случайное отклонение объекта от основного направления
        frames_count,       # количество кадров в примере
        field_size,         # размер изображения (высота, ширина)
        square_size,        # сторона квадрата (объекта)
        style               # вид тректории
):
    height, width = field_size
    dx, dy = direction
    cx, cy = start_pos
    half = square_size // 2

    for frame_i in range(frames_count):
        # Выбор смещения в зависимости от стиля движения
        if style == "noisy":
            noise_dx = np.random.randint(-noise, noise + 1)
            noise_dy = np.random.randint(-noise, noise + 1)
        elif style == "curved":
            if dx != 0:
                noise_dx = 0
                noise_dy = random.choice([-noise, noise])
            else:
                noise_dx = random.choice([-noise, noise])
                noise_dy = 0
        elif style == "impulse" and frame_i == frames_count // 2:
            noise_dx = random.choice([-3, 3])
            noise_dy = random.choice([-3, 3])
        else:
            noise_dx = 0
            noise_dy = 0

        # Обновляем координаты центра объекта
        cx_tmp = cx + dx * speed + noise_dx
        cy_tmp = cy + dy * speed + noise_dy

        # Проверяем, что квадрат полностью помещается в пределах изображения
        if cx_tmp - half < 0 or cx_tmp + half >= width:
            return False
        if cy_tmp - half < 0 or cy_tmp + half >= height:
            return False

        cx, cy = cx_tmp, cy_tmp
    return True



# Генерация датасета для обучения "камеры" различать направления движения объекта внутри кадров
def generate_dataset(
        directions=[                            # базовые направления движения объекта
            (1,0), (0,1), (-1,0), (0,-1),       # вправо, вниз, влево, вверх
            (1,1), (1,-1), (-1,1), (-1,-1)      # вправо-вниз, вправо-вверх, влево-вниз, влево-вверх
        ],
        styles=TRAJECTORY_STYLES,               # виды траекторий (определяют зашумленность базового направления)
        samples_per_style=40,                   # количество примеров конкретного стиля на каждое направление
        field_size=(28,28),                     # размер изображения (высота, ширина)
        square_size=7,                          # сторона квадрата в пикселях
        speed_var=[1,2],                        # возможные скорости движения объекта в пикселях за кадр
        noise_var=[0,1],                        # возможные значения зашумления
        frames_range=(5,8),                     # число кадров на один пример
        color_var=True,                         # если True, яркость квадрата колеблется
        save_path="data/dataset_custom.pkl"     # путь для сохранения датасета
):
    dataset = []
    height, width = field_size
    # Минимально возможный отступ от краев изображения до центра квадрата, 
    # чтобы квадрат полностью помещался на изображении
    min_start = square_size // 2 + 2  

    # Если нужно, заранее генерируем цветные квадраты
    # Иначе квадраты будут просто белыми
    squares = []
    if color_var:
        for _ in range(samples_per_style):
            squares.append(
                np.clip(
                np.random.normal(loc=1.0, scale=0.15, size=(square_size, square_size)),
                0.5, 1.0)
            )

    # Для каждого базового направления
    for d in directions:
        # Генерируем различные траектории
        for style in styles:
            # На каждую траекторию samples_per_style примеров
            count = 0
            while count < samples_per_style:
                # Матрица яркостей квадрата
                square = squares[count] if color_var else None
                # Случайная стартовая позиция объекта
                start_x = random.randint(min_start, width - min_start - 1)
                start_y = random.randint(min_start, height - min_start - 1)
                # Случайная скорость объекта в пикселях за кадр
                speed = random.choice(speed_var)
                # Случайное отклонение от основного направления
                noise = random.choice(noise_var)
                # Сколько кадров будет сгенерировано для текущей траектории
                frames_count = random.randint(frames_range[0], frames_range[1])

                # Если при выбранных параметрах движения объект не выходит за пределы кадра
                if check_trajectory(
                    direction=d,
                    start_pos=(start_x,start_y),
                    speed=speed,
                    noise=noise,
                    frames_count=frames_count,
                    field_size=field_size,
                    square_size=square_size,
                    style=style
                ):
                    # Добавляем сгенерированный пример в датасет
                    sample = generate_one_sample(direction=d,
                                                start_pos=(start_x, start_y),
                                                speed=speed,
                                                noise=noise,
                                                frames_count=frames_count,
                                                field_size=field_size,
                                                square_size=square_size,
                                                style=style,
                                                square=square
                                                )
                    dataset.append(sample)
                    count += 1

    # Сохраняем датасет как .pkl
    save_pickle(
        save_path=save_path,
        data=dataset
    )

    return len(dataset)

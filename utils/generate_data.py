import numpy as np
import random
import hashlib

from .data_converter import save_pickle



TRAJECTORY_STYLES = ["linear", "noisy", "curved", "impulse"]



# Генерация одного примера (последовательности кадров)
def generate_one_sample(
        direction,              # (dx, dy)
        start_pos,              # (cx, cy) начальная позиция центра
        speed,                  # скорость в пикселях/кадр
        noise,                  # величина случайного отклонения
        frames_count,           # число кадров в примере
        field_size=(28, 28),    # размер поля (высота, ширина)
        square_size=7,          # сторона квадрата
        style="linear",         # тип траектории
        square=None             # матрица яркостей квадрата
):
    height, width = field_size
    dx, dy = direction
    cx, cy = start_pos
    half = square_size // 2
    frames = []

    for frame_i in range(frames_count):
        frame = np.zeros(field_size, dtype=np.float32)

        # Выбор случайного смещения в зависимости от стиля
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

        # Обновляем координаты центра
        cx += dx * speed + noise_dx
        cy += dy * speed + noise_dy

        # Ограничиваем координаты, чтобы квадрат помещался в поле
        cx = np.clip(cx, half, width - half - 1)
        cy = np.clip(cy, half, height - half - 1)

        # Если квадрат не передан, создаем белый квадрат
        if square is None:
            square = np.ones((square_size, square_size), dtype=np.float32)

        x_min = int(cx - half)
        x_max = int(cx + half) + 1
        y_min = int(cy - half)
        y_max = int(cy + half) + 1

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



# Проверка выхода объекта за границы
def check_trajectory(
        direction, start_pos, speed, noise,
        frames_count, field_size, square_size,
        style
):
    height, width = field_size
    dx, dy = direction
    cx, cy = start_pos
    half = square_size // 2

    for frame_i in range(frames_count):
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

        cx_tmp = cx + dx * speed + noise_dx
        cy_tmp = cy + dy * speed + noise_dy

        # Если не помещается в поле, возвращаем False
        if (cx_tmp - half < 0) or (cx_tmp + half >= width):
            return False
        if (cy_tmp - half < 0) or (cy_tmp + half >= height):
            return False

        cx, cy = cx_tmp, cy_tmp

    return True



# Запоминаем примеры для отсечения дубликатов
def _hash_frames(frames):
    frames_array = np.stack(frames, axis=0)     # (num_frames, h, w)
    frames_byte = (frames_array * 255).astype(np.uint8).tobytes()
    h = hashlib.md5(frames_byte).hexdigest()
    return h




# Генерация полного датасета
def generate_dataset(
        directions=[(1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)],
        styles=TRAJECTORY_STYLES,
        samples_per_style=40,
        field_size=(28,28),
        square_size=7,
        speed_var=[1,2],
        noise_var=[0,1],
        color_var=True,
        frames_count=7,
        max_tries=10000,
        save_path="data/dataset_custom.pkl"
):
    dataset = []
    height, width = field_size
    half = square_size // 2
    min_start = half + 2

    squares = []
    if color_var:
        total_squares_needed = samples_per_style * len(styles)
        for _ in range(total_squares_needed):
            sq = np.clip(
                np.random.normal(loc=1.0, scale=0.15, size=(square_size, square_size)),
                0.5, 1.0
            )
            squares.append(sq)
    else:
        squares = [None] * (samples_per_style * len(styles))

    seen_trajectories = set()

    total_needed = len(directions) * len(styles) * samples_per_style
    added_count = 0
    tries_count = 0

    while added_count < total_needed and tries_count < max_tries:
        tries_count += 1

        d = random.choice(directions)
        style = random.choice(styles)
        speed = random.choice(speed_var)
        noise = random.choice(noise_var)

        chosen_square = random.choice(squares)

        start_x = random.randint(min_start, width - min_start - 1)
        start_y = random.randint(min_start, height - min_start - 1)

        is_valid = check_trajectory(
            direction=d,
            start_pos=(start_x, start_y),
            speed=speed,
            noise=noise,
            frames_count=frames_count,
            field_size=field_size,
            square_size=square_size,
            style=style
        )

        if not is_valid:
            continue

        sample = generate_one_sample(
            direction=d,
            start_pos=(start_x, start_y),
            speed=speed,
            noise=noise,
            frames_count=frames_count,
            field_size=field_size,
            square_size=square_size,
            style=style,
            square=chosen_square
        )

        frames = sample["frames"]

        diff_sum = np.sum(np.abs(frames[-1] - frames[0]))
        if diff_sum < 1.0:
            continue

        traj_hash = _hash_frames(frames)
        if traj_hash in seen_trajectories:
            continue

        seen_trajectories.add(traj_hash)
        dataset.append(sample)
        added_count += 1

    save_pickle(save_path=save_path, data=dataset)
    return len(dataset)

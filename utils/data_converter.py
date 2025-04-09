import pickle
import os
import numpy as np
from PIL import Image

from core.input_layer import init_event_generator, generate_events
from utils.visualization import plot_events




# Сохранение объекта data в файл .pkl по пути save_path
def save_pickle(save_path, data):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)



# Загрузка объекта из файла .pkl
def load_pickle(load_path="data/dataset_custom.pkl"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Файл {load_path} не найден")
    with open(load_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data



# Получение изображения из массива нормализованных яркостей
def arr_to_image(image_arr, save_path=None):
    image = image_arr * 255.0
    image = image.astype(np.uint8)
    img = Image.fromarray(image)
    img.show()
    if save_path is not None:
        img.save(save_path)



# Выборочная проверка полученных изображений
def dataset_dict_to_image(dataset_path, num_ex=1):
    dataset_dict = load_pickle(load_path=dataset_path)
    len_dataset = len(dataset_dict)
    for _ in range(num_ex):
        sample_index = np.random.randint(0, len_dataset + 1)
        sample = dataset_dict[sample_index]
        frames = sample["frames"]
        for frame in frames:
            arr_to_image(frame)



# Проверка генерации событий на датасете
def dataset_dict_to_events(dataset_path, num_ex=1):
    dataset_dict = load_pickle(load_path=dataset_path)
    len_dataset = len(dataset_dict)
    for _ in range(num_ex):
        sample_index = np.random.randint(0, len_dataset + 1)
        sample = dataset_dict[sample_index]
        frames = sample["frames"]
        num_frames = len(frames)
        ev_generator = init_event_generator()
        old_frame = frames[0]
        cur_t = 0
        dt = 33
        cur_events = []
        for idx in range(1, num_frames):
            new_frame = frames[idx]
            ev = generate_events(
                state=ev_generator,
                old_frame=old_frame,
                new_frame=new_frame,
                prev_t=cur_t,
                new_t=cur_t+dt
            )
            cur_t += dt
            cur_events.extend(ev)
        plot_events(cur_events)





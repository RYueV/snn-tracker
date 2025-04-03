import numpy as np


# Класс для моделирования движения объекта (квадрат со стороной 1 + 2*obj_radius) по сцене
class Moving_object:
    def __init__(
            self,
            field_size=(64,64),     # размер сцены (высота, ширина в пикселях)
            obj_radius=1            # радиус объекта (в пикселях)
    ):
        # Размеры поля
        self.height, self.width = field_size  
        # Половина стороны квадрата
        self.obj_radius = obj_radius            

        # x, y - текущие координаты объекта; изначально объект в центре поля
        self.x = self.width // 2
        self.y = self.height // 2

        # Направление движения (значение смещения в пикселях)
        # По ширине: если > 0, то вправо, если < 0, то влево
        self.dir_x = 0
        # По высоте: если > 0, то вниз, если < 0, то вверх
        self.dir_y = 0


    # Сброс состояния объекта
    def reset(
            self,
            start_pos=None,         # стартовая позиция (кортеж: сначала по x, потом по y)
            direction=(0,0)         # направление движения (по умолчанию объект стоит на месте)
    ):
        # Если новая стартовая позиция не указана явно, то перемещаем объект в центр поля
        if start_pos:
            self.x, self.y = start_pos
        else:
            self.x, self.y = self.width // 2, self.height // 2
        self.dir_x, self.dir_y = direction



    # Один шаг движения объекта
    def step(
            self,
            noise=1         # максимально возможное отклонение от основной траектории
    ):
        """

        Основная траектория движения задается через (dir_x, dir_y).
        На каждом шаге к основной траектории добавляется случайное смещение [-noise;+noise]

        """
        # Определяем смещение
        noise_dx = np.random.randint(-noise, noise + 1)
        noise_dy = np.random.randint(-noise, noise + 1)

        # Вычисляем новую позицию
        new_x = self.x + self.dir_x + noise_dx
        new_y = self.y + self.dir_y + noise_dy

        # Ограничиваем координаты, чтобы не выйти за границы поля
        self.x = np.clip(new_x, 0, self.width - 1)
        self.y = np.clip(new_y, 0, self.height - 1)


    # Установка нового направления движения
    def set_base_direction(self, dx, dy):
        self.dir_x = dx
        self.dir_y = dy


    # Генерирует текущую сцену - черное поле с белым объектом (квадрат)
    def show_scene(self):
        # Инициализируем поле нулями (все пиксели черные)
        frame = np.zeros((self.height, self.width), dtype=np.float32)

        # Обходим квадратную область вокруг центра (в пределах half_size)
        for dy in range(-self.obj_radius, self.obj_radius + 1):
            for dx in range(-self.obj_radius, self.obj_radius + 1):
                xx = np.clip(self.x + dx, 0, self.width - 1)
                yy = np.clip(self.y + dy, 0, self.height - 1)
                # Яркость объекта
                frame[yy, xx] = 1.0  

        return frame

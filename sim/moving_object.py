import numpy as np


# Класс для моделирования движения объекта (квадрат со стороной 1 + 2*obj_radius) по сцене
class Moving_Object:
    def __init__(
            self,
            field_size=(64,64),     # размер сцены (высота, ширина в пикселях)
            obj_radius=2,           # радиус объекта (в пикселях)
            start_x=None,           # начальная координата по ширине
            start_y=None,           # начальная координата по высоте
            direction=(1,1),        # направление движения (dir_x, dir_y)
            noise=0                 # максимальное отклонение от траектории
    ):
        self.field_height, self.field_width = field_size  
        self.dir_x, self.dir_y = direction
        self.obj_radius = obj_radius    
        self.noise = noise

        if start_x is None:
            start_x = self.field_width // 2
        if start_y is None:
            start_y = self.field_height // 2

        # x, y - текущие координаты центра объекта
        self.center_x = int(start_x)
        self.center_y = int(start_y)



    # Один шаг движения объекта
    def step(self):
        """

        Основная траектория движения задается через (dir_x, dir_y).
        На каждом шаге к основной траектории добавляется случайное смещение [-noise;+noise]

        """
        # Определяем смещение
        noise_dx = np.random.randint(-self.noise, self.noise + 1)
        noise_dy = np.random.randint(-self.noise, self.noise + 1)
        # Вычисляем новую позицию
        new_x = self.center_x + self.dir_x + noise_dx
        new_y = self.center_y + self.dir_y + noise_dy
        # Ограничиваем координаты, чтобы не выйти за границы поля
        new_x = np.clip(new_x, self.obj_radius, self.field_width - self.obj_radius)
        new_y = np.clip(new_y, self.obj_radius, self.field_height - self.obj_radius)
        # Обновляем координаты центра
        self.center_x = new_x
        self.center_y = new_y



    # "Рисует" белый квадрат на кадре frame 
    # (frame - numpy матрица с нормированными значениями яркости)
    def fix_obj(self, frame):
        x_min = int(self.center_x - self.obj_radius)
        x_max = int(self.center_x + self.obj_radius)
        y_min = int(self.center_y - self.obj_radius)
        y_max = int(self.center_y + self.obj_radius)
        frame[y_min:y_max+1, x_min:x_max+1] = 1.0


    


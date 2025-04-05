import numpy as np

from .moving_object import Moving_Object
from .camera import Camera


# Симуляция слежения
class Tracking_Object:
    def __init__(
        self,
        field_size=(64,64),     # размер поля
        window_size=(28,28),    # размер окна обзора камеры
        obj_radius=2,           # половина стороны квадрата
        obj_direction=(1,0),    # направление движения (dir_x, dir_y)
        noise=0                 # максимальное отклонение от траектории
    ):
        # Размер поля
        self.field_width, self.field_height = field_size

        # Объект (белый квадрат со стороной 1+2*obj_radius)
        self.object = Moving_Object(
            field_size=field_size,
            obj_radius=obj_radius,
            direction=obj_direction,
            noise=noise
        )

        # Камера
        self.camera = Camera(
            field_size=field_size,
            window_size=window_size
        )

        # Текущая сцена (обновляется на каждом шаге)
        self.current_field = np.zeros((self.field_height, self.field_width), dtype=float)


    # Сброс симуляции
    def reset(self):
        self.object = Moving_Object(
            field_size=(self.field_width, self.field_height),
            obj_radius=self.object.obj_radius,
            direction=(self.object.dir_x, self.object.dir_y),
            noise=self.object.noise
        )
        self.camera = Camera(
            field_size=(self.field_width, self.field_height),
            window_size=(self.camera.window_width, self.camera.window_height)
        )
        self.current_field = np.zeros((self.field_height, self.field_width), dtype=float)


    # Шаг симуляции
    def step(self):
        # Двигаем объект
        self.object.step()
        # Двигаем камеру (пытаемся центрировать объект)
        self._follow_object()
        # Генерируем картинку всей сцены
        self.current_field[:] = 0.0  
        self.object.fix_obj(self.current_field)  


    # Простое слежение за объектом (без snn)
    def _follow_object(self):
        # Центр окна камеры
        cam_cx = self.camera.top_left_x + self.camera.window_width // 2
        cam_cy = self.camera.top_left_y + self.camera.window_height // 2

        # Расстояние от объекта до центра камеры
        dx = self.object.center_x - cam_cx
        dy = self.object.center_y - cam_cy

        # Двигаемся на шаг к объекту
        move_x, move_y = 0, 0

        # Если объект слишком сильно ушел влево/вправо
        if abs(dx) > 2:  
            move_x = int(np.sign(dx))

        # Если объект слишком сильно ушел вверх/вниз
        if abs(dy) > 2:
            move_y = int(np.sign(dy))

        # Двигаем камеру
        self.camera.step(move_x, move_y)


    # Возвращает окно, которое "видит" камера
    def get_camera_view(self):
        return self.camera.get_view(self.current_field)





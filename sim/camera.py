import numpy as np

class Camera:
    def __init__(
            self,
            field_size=(64,64),     # размер поля (высота, ширина)
            window_size=(28,28)     # размер окна обзора камеры (высота, ширина)
    ):
        self.field_height, self.field_width = field_size
        self.window_height, self.window_width = window_size
        # Координаты верхнего левого угла камеры
        self.x = (self.field_width - self.window_width) // 2
        self.y = (self.field_height - self.window_height) // 2


    # Перемещение окна камеры по полю на заданное количество пикселей
    def step(self, dx, dy):
        # Обновляем координаты верхнего левого угла, учитывая ограничения на размеры камеры и поля
        self.x = np.clip(self.x + dx, 0, self.field_width - self.window_width)
        self.y = np.clip(self.y + dy, 0, self.field_height - self.window_height)


    # Обзор камеры
    def get_view(
            self,
            field       # матрица всей сцены
    ):
        # Верхняя и нижняя границы окна по вертикали
        y_top = self.y
        y_bottom = self.y + self.window_height

        # Левая и правая границы окна по горизонтали
        x_left = self.x
        x_right = self.x + self.window_width

        # Берем только ту часть поля, которую камера видит
        camera_view = field[y_top:y_bottom, x_left:x_right]

        return camera_view



    def get_center_offset(self, object_x, object_y):
        """Считает смещение объекта от центра камеры"""
        cam_center_x = self.x + self.window_width // 2
        cam_center_y = self.y + self.window_height // 2
        return object_x - cam_center_x, object_y - cam_center_y

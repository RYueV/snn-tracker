import numpy as np

class Camera:
    def __init__(
            self,
            field_size=(64,64),     # размер поля (высота, ширина)
            window_size=(28,28),    # размер окна обзора камеры (высота, ширина)
            start_x=None,
            start_y=None
    ):
        self.field_height, self.field_width = field_size
        self.window_height, self.window_width = window_size

        # Координаты верхнего левого угла камеры
        if start_x is None:
            start_x = (self.field_width - self.window_width) // 2
        if start_y is None:
            start_y = (self.field_height - self.window_height) // 2
        self.top_left_x = int(start_x)
        self.top_left_y = int(start_y)


    # Перемещение окна камеры по полю на заданное количество пикселей
    def step(self, dx, dy):
        # Обновляем координаты верхнего левого угла, учитывая ограничения на размеры камеры и поля
        self.top_left_x = np.clip(self.top_left_x + dx, 0, self.field_width - self.window_width)
        self.top_left_y = np.clip(self.top_left_y + dy, 0, self.field_height - self.window_height)


    # Обзор камеры
    def get_view(
            self,
            frame       # матрица всей сцены (кадр)
    ):
        # Верхняя и нижняя границы окна по вертикали
        y_top = self.top_left_y
        y_bottom = self.top_left_y + self.window_height

        # Левая и правая границы окна по горизонтали
        x_left = self.top_left_x
        x_right = self.top_left_x + self.window_width

        # Берем только ту часть поля, которую камера видит
        return frame[y_top:y_bottom, x_left:x_right]




import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation


from .tracking_object import Tracking_Object


def simulate(
        steps=100,                  # количество шагов симуляции
        interval_ms=200,            # интервал между кадрами (мс)
        field_size=(80,80),         # размер всего поля (ширина и высота в пикселях)
        window_size=(28,28),        # обзор камеры (ширина и высота в пикселях)
        obj_radius=1,               # половина стороны квадрата (объект)
        obj_direction=(1,0),        # направление движения
        noise=0                     # максимальное отклонение от основной траектории
):
    # Настриваем симуляцию
    simulator = Tracking_Object(
        field_size=field_size,
        window_size=window_size,
        obj_radius=obj_radius,
        obj_direction=obj_direction,
        noise=noise
    )
    simulator.reset()

    fig, ax = plt.subplots()

    # Изначальная картинка
    img = ax.imshow(simulator.current_field, cmap='gray', vmin=0, vmax=1)
    ax.set_title("Tracking simulation")
    ax.axis("off")

    # Прямоугольник, отражающий положение камеры
    rect = Rectangle(
        (simulator.camera.top_left_x, simulator.camera.top_left_y),
        simulator.camera.window_width,
        simulator.camera.window_height,
        linewidth=1.5,                      # толщина рамки
        edgecolor='red',                    # рамка красного цвета
        facecolor='none'                    # заливки нет
    )
    ax.add_patch(rect)


    # Функция изменения кадра
    def update(frame):
        # Обновляем симуляцию на один шаг
        simulator.step()
        # Обновляем картинку
        img.set_array(simulator.current_field)
        # Обновляем положение прямоугольника
        rect.set_xy((simulator.camera.top_left_x, simulator.camera.top_left_y))
        return [img, rect]


    a = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval_ms,
        blit=True
    )

    plt.show()
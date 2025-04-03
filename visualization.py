import matplotlib.pyplot as plt
import matplotlib.animation as animation


from sim import Moving_object


##################   Отображение траектории объекта   #####################


def show_moving_obj(
        steps=100,              # количество шагов движения
        noise=1,                # максимальная амплитуда случайного отклонения
        direction=(1,1)         # направление движения объекта
):
    # Создаем объект
    obj = Moving_object()
    obj.reset(direction=direction)

    # Фигура и ось для анимации
    fig, ax = plt.subplots()
    image = ax.imshow(obj.show_scene(), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Траектория объекта")
    ax.axis('off')

    # Функция обновления кадра
    def update(frame_num):
        obj.step(noise=noise)
        new_frame = obj.show_scene()
        image.set_array(new_frame)
        return [image]

    # Создаем анимацию
    a = animation.FuncAnimation(
        fig,                # фигура, на которой все рисуется
        update,             # функция обновления объекта
        frames=steps,       # количество кадров (шагов перемещения)
        interval=100,       # интервал времени между кадрами (мс)
        blit=True,          # ускорение отрисовки
        repeat=False        # повторять ли анимацию после завершения
    )

    plt.tight_layout()
    plt.show()

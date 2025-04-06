import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

from .tracking_object import Tracking_Object
from core.input_layer import init_event_generator, generate_events
import utils.visualization as v



def simulate(
        steps=60,                   # количество шагов симуляции
        observe_steps=10,           # первые observe_steps шагов камера не двигается
        interval_ms=100,            # интервал между кадрами в анимации (мс)
        dt=33,                      # время между кадрами в симуляции (мс)
        field_size=(80, 80),        # размер всего поля (ширина и высота в пикселях)
        window_size=(28, 28),       # обзор камеры (ширина и высота в пикселях)
        obj_radius=2,               # половина стороны квадрата (объект)
        obj_direction=(1, 0),       # направление движения объекта (x, y)
        noise=0,                    # максимальное отклонение от основной траектории
        pixel_ref=3.0,              # рефрактерный период события
        threshold=0.3,              # логарифмический порог изменения яркости для генерации события
        show_hist=True              # строить ли гистограмму после периода наблюдения
):
    """
    Основная функция для симуляции задачи трекинга квадрата и генерации событий DVS-подобным способом.

    Логика:
      1) Создаётся объект трекинга, который внутри себя содержит Moving_Object и Camera.
         При reset() инициализируются объект и камера в центре поля (так что квадрат будет в центре камеры).
      2) Первые observe_steps кадров: объект двигается, а камера стоит на месте.
      3) После observe_steps: камера начинает "автоматически" следовать за объектом
         (метод simulator.step() включает в себя и шаг объекта, и шаг камеры).
      4) Между старым и новым кадром (old_frame, new_frame) генерируются события через generate_events.
         Время для событий интерполируется от prev_t до new_t (каждый шаг увеличиваем t на dt).
      5) Визуализируем все кадры в анимации через matplotlib.
         Гистограмму (и scatter-графики) событий показываем единожды в момент перехода через observe_steps.

    """

    # Настриваем симуляцию (камера и объект)
    simulator = Tracking_Object(
        field_size=field_size,
        window_size=window_size,
        obj_radius=obj_radius,
        obj_direction=obj_direction,
        noise=noise
    )
    # Сбрасываем в начальное состояние (объект и камера в центре)
    simulator.reset()

    # Инициализируем генератор событий (генерируем словарь состояния генератора)
    state = init_event_generator(frame_shape=window_size, pixel_ref=pixel_ref)
    # Список событий
    all_events = []     
    # Текущее время в симуляции
    cur_time = 0.0      

    # Берем первый кадр из камеры
    prev_view = simulator.get_camera_view()

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
        linewidth=1.5,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Флаг, чтобы гистограмму вызвать только один раз
    hist_shown = False

    # Функция изменения кадра
    def update(frame_index):
        nonlocal cur_time, prev_view, state, all_events, hist_shown

        # Если период наблюдения не закончился, то двигаем только объект
        if frame_index < observe_steps:
            simulator.object.step()
            simulator.current_field[:] = 0.0
            simulator.object.fix_obj(simulator.current_field)
        # Иначе двигаем и объект, и камеру
        else:
            simulator.step()

        # Обновляем картинку поля
        img.set_array(simulator.current_field)
        rect.set_xy((simulator.camera.top_left_x, simulator.camera.top_left_y))

        # Получаем новый кадр из камеры
        new_view = simulator.get_camera_view()

        # Генерация событий между предыдущим и новым кадром
        events = generate_events(
            state=state,
            old_frame=prev_view,
            new_frame=new_view,
            prev_t=cur_time,
            new_t=cur_time + dt,
            threshold=threshold
        )
        all_events.extend(events)

        # Обновляем кадр и время
        prev_view = new_view.copy()
        cur_time += dt

        if show_hist and (frame_index == observe_steps) and not hist_shown:
            v.plot_events(all_events, dt=dt)
            hist_shown = True

        # Возвращаем объекты, которые нужно перерисовывать анимации
        return [img, rect]


    a = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval_ms,  
        blit=True
    )

    plt.show()

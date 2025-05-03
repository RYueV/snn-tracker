import random
from core.global_config import FRAME_DT_MS


# Генерация словаря гиперпараметров скрытого слоя
def generate_params():
    """
    Зависимости между параметрами:
    1) t_ltp - верхняя граница значения разности между временем поступления
       входного сигнала и временем спайка нейрона, при котором мы можем считать,
       что входной сигнал пришел "недавно";
       t_ref - время, на которое "выключается" нейрон после спайка;
       t_inhibit - время, на которое сработавший нейрон "глушит" остальные;
       tau_leak - постоянная времени утечки потенциала;
                  определяет, насколько долго нейрон помнит предыдущие входы;
                  если tau_leak слишком маленькая, то входной потенциал не накопится до порога;
                  если слишком большая, то сигнал от старых входов будет влиять слишком долго.

    2) Константы времени задаются в мс исходя из того, что при текущих настройках
       между двумя соседними кадрами проходит 16.7 мс

    3) t_inhibit < t_ref, чтобы дать возможность нескольким нейронам среагировать на
       одну и ту же пару (направление, картинка).

    4) t_inhibit < t_ltp и t_ref < t_ltp, чтобы нейроны успевали отреагировать на входы.
       
    5) tau_leak должно быть немного больше, чем t_ltp, чтобы входы, поступившие во время
       обучения на одной паре (направление, картинка), не забывались.

    6) Начальные веса связей (w_init) задаются нормальным распределением 
       со средним w_init_mean и дисперсией w_init_std.

    7) Пороговое значение потенциала нейрона для генерации спайка I_thres
       должно быть таким, чтобы при выбранных значениях STDP нейрон успевал
       спайковать ожидаемое количество раз.
    """
    # Временные константы LIF и STDP
    t_ltp_min = 0.6 * FRAME_DT_MS
    t_ltp_max = 1.1 * FRAME_DT_MS
    t_ltp = random.uniform(t_ltp_min, t_ltp_max)
    t_inhibit = random.uniform(0.5, min(5.0, t_ltp - 4.0))
    t_ref = random.uniform(t_inhibit + 1.0, min(10.0, t_ltp - 2.0))
    tau_leak = random.uniform(t_ltp + 10.0, t_ltp + 25.0)

    # Порог спайка и коэффициенты STDP
    I_thres = random.uniform(450.0, 600.0)
    alpha_plus = random.uniform(6.0, 10.0)
    alpha_minus = random.uniform(5.0, 9.0)
    beta_plus = random.uniform(0.08, 0.18)
    beta_minus = random.uniform(0.08, 0.18)

    # Распределение начальных весов
    w_init_mean = random.uniform(60.0, 80.0)
    w_init_std  = random.uniform(15.0, 25.0)
    w_min, w_max = 20.0, 500.0     

    return {
        "TAU_LEAK": tau_leak,
        "I_THRES": I_thres,
        "T_REF": t_ref,
        "T_INHIBIT": t_inhibit,
        "ALPHA_PLUS": alpha_plus,
        "ALPHA_MINUS": alpha_minus,
        "BETA_PLUS": beta_plus,
        "BETA_MINUS": beta_minus,
        "T_LTP": t_ltp,
        "W_INIT_MEAN": w_init_mean,
        "W_INIT_STD": w_init_std,
        "W_MIN": w_min,
        "W_MAX": w_max
    }



# Получение словаря параметров потомка из словарей родителей
def mix_params(p1, p2):
    """
    p1, p2: словари с параметрами STDP, LIF, SNN
    """
    # Инициализируем новый словарь
    child = {}

    # Идем по ключам (наименования параметров)
    for key in p1.keys():
        # Извлекаем значения параметров родителей
        param_p1 = p1[key]
        param_p2 = p2[key]
        # Присваиваем потомку значение по ключу 
        # (равновероятный выбор)
        child[key] = (param_p1 if random.random() < 0.5 else param_p2)

    return child



# Получение нового словаря из старого путем мутаций (гауссово распределение)
def mutate_params(
        parent,             # исходный словарь параметров
        mutation_prob=0.2,  # вероятность мутации
        sigma=0.1           # e^(N(0, sigma^2))
):
    # Создаем копию исходного словаря
    child = dict(parent)

    # Идем по ключам (наименования параметров)
    for key in parent:
        # Предельные значения весов зафиксированы, их пропускаем
        if key in ("W_MIN", "W_MAX"):
            continue
        # Решаем, нужна ли мутация
        if random.random() < mutation_prob:
            # Подбираем коэффициент изменения значения параметра
            child[key] *= random.gauss(1.0, sigma)

    return child



# Отбор лучших кандидатов поколения
def candidate_selection(
        candidates,         # список кандидатов в формате (энтропия, словарь параметров)
        group_size=3        # сколько случайных кандидатов сравниваем
):
    # Выбираем случайную группу из всех кандидатов
    group = random.sample(candidates, group_size)
    # Сортируем подгруппу по энтропии (от лучшего к худшему)
    group.sort(key=lambda x: x[0])
    # Возвращаем лучшего кандидата
    return group[0][1]


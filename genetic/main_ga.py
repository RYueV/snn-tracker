import random
import genetic.ga_config as ga
from .operators import (
    generate_params,
    mix_params,
    mutate_params,
    candidate_selection
)
from .train_snn import evaluate_selectivity
from core.global_config import COUNT_NEURONS
from utils.visualization import plot_direction_heatmap


# Фиксируем рандом
random.seed(42)



# Формирование первого поколения
def init_population():
    population = []
    print(f"### Поколение 1/{ga.GENERATIONS} ###")
    for num_child in range(ga.POP_SIZE):
        # Генерируем набор параметров
        params = generate_params()
        # Обучаем скрытый слой на этом наборе и оцениваем качество
        anti_selectivity_score, spike_matrix = evaluate_selectivity(
            params=params,
            distr_penalty=0.3,
            dataset=ga.DATASET
        )
        population.append((anti_selectivity_score, params))
        print(f"\tИндивид {num_child+1}/{ga.POP_SIZE}: anti_selectivity_score = {anti_selectivity_score}")
        ##### Визуализация и запись нужны только для отладки #####
        spikes_count_by_dir = {}
        for dir_idx, direction in enumerate(ga.DIR2IDX.keys()):
            spikes_count_by_dir[direction] = spike_matrix[:, dir_idx]
        #plot_direction_heatmap(spikes_count_by_dir, list(ga.DIR2IDX.keys()), COUNT_NEURONS)

    # Сортируем в порядке от лучшего к худшему
    population.sort(key=lambda x: x[0])
    return population



# Главная функция ГА
def genetic_search():
    # Формируем первое поколение
    cur_population = init_population()
    # Следующие частично получаем путем изменения параметров первого
    for gen in range(1, ga.GENERATIONS + 1):
        print(f"### Поколение {gen+1}/{ga.GENERATIONS} ###")
        next_population = []
        # Несколько лучших особей переходят в следующее поколение без изменений
        next_population.extend(cur_population[:ga.NUM_BEST_INDIV])
        # Создаем остальных потомков
        num_child = ga.NUM_BEST_INDIV
        while len(next_population) < ga.POP_SIZE:
            num_child += 1
            p1 = candidate_selection(
                candidates=cur_population,
                group_size=3
            )
            p2 = candidate_selection(
                candidates=cur_population,
                group_size=3
            )
            child = mix_params(p1, p2)
            child = mutate_params(
                parent=child,
                mutation_prob=ga.MUTATION_PROB,
                sigma=0.1
            )
            # Обучаем скрытый слой на этом наборе и оцениваем качество
            anti_selectivity_score, spike_matrix = evaluate_selectivity(
                params=child,
                distr_penalty=0.3,
                dataset=ga.DATASET
            )
            print(f"\tИндивид {num_child}/{ga.POP_SIZE}: anti_selectivity_score = {anti_selectivity_score}")
            next_population.append((anti_selectivity_score , child))

            ##### Визуализация и запись нужны только для отладки #####
            spikes_count_by_dir = {}
            for dir_idx, direction in enumerate(ga.DIR2IDX.keys()):
                spikes_count_by_dir[direction] = spike_matrix[:, dir_idx]
            #plot_direction_heatmap(spikes_count_by_dir, list(ga.DIR2IDX.keys()), COUNT_NEURONS)
            with open("output.txt", "a") as f:
                f.write(f"[GEN {gen:02}] individual {len(next_population)}:\n")
                f.write(f"anti_selectivity_score = {anti_selectivity_score:.4f}\n")
                for key, val in child.items():
                    f.write(f" {key}: {val:.5f}\n")
                f.write("\n")
            #####

        # Сортируем от лучшего к худшему
        next_population.sort(key=lambda x: x[0])
        # Переходим к следующему поколению
        cur_population = next_population
        # Запоминаем лучший результат в текущем поколении
        best_score, best_params = cur_population[0]
        print(f"[GEN {gen:02}]  best score = {best_score:.4f}")

    # Финальный результат
    best_score, best_params = cur_population[0]
    print(f"best score={best_score}\n{best_params}")




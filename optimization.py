import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from main import run_simulation, Q_normal, r_normal, Q_drought, r_drought, Q_rainy, r_rainy


def genetic_algorithm(num_generations, population_size, elite_size=0.05):
    elite_count = int(elite_size * population_size)
    population = [np.random.uniform(low=[0.5, 0.05, 0.01, 0.4, 0.4, 0.2, 0.2, 0.2, 0.4, 0.4, 0.2],
                                    high=[2.0, 0.2, 0.1, 0.6, 0.6, 0.8, 0.8, 0.8, 0.6, 0.6, 0.8], size=11) for _ in
                  range(population_size)]
    best_fitness_scores = []

    mutation_rate = 0.1
    t = tqdm(range(num_generations), desc="Genetic optimization", unit="generation")
    for _ in t:
        fitness_scores = []
        for individual in population:
            kp1, ki1, kd1, setpoint1, setpoint2, weight_v1, weight_v2, weight_combined, kp2, ki2, kd2 = individual
            pid_params = [kp1, ki1, kd1, kp2, ki2, kd2, setpoint1, setpoint2, weight_v1, weight_v2, weight_combined]

            valid_count_normal, low_count_normal, high_count_normal, invalid_count_normal = run_simulation(
                Q_normal, r_normal, "Normal Weather", params=pid_params, optimize=True)
            valid_count_drought, low_count_drought, high_count_drought, invalid_count_drought = run_simulation(
                Q_drought, r_drought, "Drought Weather", params=pid_params, optimize=True)
            valid_count_rainy, low_count_rainy, high_count_rainy, invalid_count_rainy = run_simulation(
                Q_rainy, r_rainy, "Rainy Weather", params=pid_params, optimize=True)

            fitness_normal = valid_count_normal - (low_count_normal + high_count_normal + invalid_count_normal)
            fitness_drought = valid_count_drought - (low_count_drought + high_count_drought + invalid_count_drought)
            fitness_rainy = valid_count_rainy - (low_count_rainy + high_count_rainy + invalid_count_rainy)

            fitness = (fitness_normal + fitness_drought + fitness_rainy) / 3
            fitness_scores.append(fitness)

        sorted_population = [x for _, x in
                             sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        selected_population = [x for x in sorted_population[:population_size // 2]]

        elites = sorted_population[:elite_count]

        offspring = []
        for _ in range(population_size - elite_count):
            parent1, parent2 = random.sample(selected_population, 2)
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child + np.random.normal(0, mutation_rate, 11))

        population = elites + offspring
        best_fitness_scores.append(max(fitness_scores))
        t.set_description(f'Fitness: {round(max(fitness_scores), 2)}')

    plt.plot(best_fitness_scores)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title('Best Fitness Score Over Generations')
    plt.grid(True)
    plt.show()

    best_params = sorted_population[0]
    return best_params

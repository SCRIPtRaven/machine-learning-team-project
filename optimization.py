import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from main import run_simulation, Q_normal, r_normal, Q_drought, r_drought, Q_rainy, r_rainy


def genetic_algorithm(num_generations, population_size, elite_size=0.05, scenario="normal"):
    elite_count = int(elite_size * population_size)

    # Define different ranges for initial parameters based on the scenario
    if scenario == "normal":
        low = [0.05, 0.005, 0.0025, 0.05, 0.1, 0.01, 0.05, 0.1]
        high = [50.0, 1.0, 1.0, 4.0, 4.0, 5.0, 5.0, 10.0]
    elif scenario == "drought":
        low = [0.1, 0.01, 0.005, 0.3, 0.3, 0.1, 0.1, 0.3]
        high = [1.0, 0.1, 0.05, 0.5, 0.5, 0.5, 0.5, 1.0]
    elif scenario == "rainy":
        low = [0.05, 0.005, 0.0025, 0.05, 0.1, 0.01, 0.05, 0.1]
        high = [100.0, 1.0, 1.0, 4.0, 4.0, 5.0, 5.0, 10.0]
    else:
        raise ValueError("Unknown scenario")

    population = [np.random.uniform(low=low, high=high, size=8) for _ in range(population_size)]
    best_fitness_scores = []

    mutation_rate = 0.1
    elite_mutation_rate = 0.05
    t = tqdm(range(num_generations), desc="Genetic optimization", unit="generation")
    for _ in t:
        fitness_scores = []
        for individual in population:
            kp, ki, kd, setpoint1, setpoint2, weight_v1, weight_v2, feedforward_gain = individual
            pid_params = [kp, ki, kd, setpoint1, setpoint2, weight_v1, weight_v2, feedforward_gain]

            if scenario == "normal":
                valid_count, low_count, high_count, invalid_count, stability_q1, stability_V1, stability_V2 = run_simulation(
                    Q_normal, r_normal, "Normal Weather", params=pid_params, optimize=True)
            elif scenario == "drought":
                valid_count, low_count, high_count, invalid_count, stability_q1, stability_V1, stability_V2 = run_simulation(
                    Q_drought, r_drought, "Drought Weather", params=pid_params, optimize=True)
            elif scenario == "rainy":
                valid_count, low_count, high_count, invalid_count, stability_q1, stability_V1, stability_V2 = run_simulation(
                    Q_rainy, r_rainy, "Rainy Weather", params=pid_params, optimize=True)
            else:
                raise ValueError("Unknown scenario")

            # Fitness calculation incorporating stability
            fitness = (valid_count - (
                    low_count + high_count + invalid_count)) + stability_q1 + stability_V1 + stability_V2
            fitness_scores.append(fitness)

        sorted_population = [x for _, x in
                             sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        selected_population = [x for x in sorted_population[:population_size // 2]]

        elites = sorted_population[:elite_count]

        offspring = []

        # Create offspring from elites
        for _ in range(elite_count):
            parent1, parent2 = random.sample(elites, 2)
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Apply mutation to only one parent
            if random.random() < 0.5:
                child += np.random.normal(0, elite_mutation_rate, 8)

            offspring.append(child)

        # Create offspring from selected population
        for _ in range(population_size - elite_count):
            parent1, parent2 = random.sample(selected_population + elites, 2)  # Allow elites to be selected as parents
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child += np.random.normal(0, mutation_rate, 8)  # Apply mutation

            offspring.append(child)

        population = elites + offspring
        best_fitness_scores.append(max(fitness_scores))
        t.set_description(f'Fitness: {round(max(fitness_scores), 2)}')

    plt.plot(best_fitness_scores)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title(f'Best Fitness Score Over Generations ({scenario.capitalize()} Weather)')
    plt.grid(True)
    plt.show()

    best_params = sorted_population[0]
    return best_params

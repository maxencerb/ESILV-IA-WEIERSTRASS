import numpy as np
from population import total_fitness, select_parents, selection
from individu import crossover
from mutation import b_c_flip_mutation, reset_mutation, a_mutation
from init import BORNE_A, BORNE_B, BORNE_C

def single_fit(population, time, temperature, number_of_parents = 20, alpha_a_crossover = .3, number_of_bc_flips = 5, number_of_resets = 10, number_of_a_variations = 10, a_std_mutation = .2, fitness = None, borne_a = BORNE_A, borne_b = BORNE_B, borne_c = BORNE_C, a_precision = 2):
    """
    Makes a Single fit over the population
    """
    # Fitness
    if np.any(fitness):
        fitness = total_fitness(population, time, temperature)
    # Parent selection
    parents = select_parents(population, fitness, number_of_parents)
    # Crossover
    childrens = []
    for couple in parents:
        p1, p2 = couple
        c1, c2 = crossover(p1, p2, alpha_a_crossover, borne_a, a_precision)
        childrens.append(c1)
        childrens.append(c2)
    childrens = np.array(childrens)
    # mutation
    if number_of_bc_flips > 0:
        b_c_flip_mutation(childrens, borne_b, borne_c, number_of_bc_flips)
    if number_of_resets > 0:
        reset_mutation(childrens, borne_a, borne_b, borne_c, number_of_resets, a_precision)
    if number_of_a_variations > 0:
        a_mutation(childrens, borne_a, number_of_a_variations, a_std_mutation, a_precision)
    # Selection
    selection(population, childrens, fitness)

def stopAfter(after = 50):
    def stop(best_fitness):
        if len(best_fitness) > after and best_fitness[-after] == best_fitness[-1]:
            return True
        return False
    return stop

import progressbar
import time as t

def fit(population, time, temperature, nb_of_cycle = 10, stopFunction = stopAfter(), number_of_parents = 20, alpha_a_crossover = .3, number_of_bc_flips = 5, number_of_resets = 10, number_of_a_variations = 10, a_std_mutation = .2, borne_a = BORNE_A, borne_b = BORNE_B, borne_c = BORNE_C, abs_error = .2, a_precision = 2):
    """
    Fit the entire population for a number of cycle with a stop function
    The stop function takes the best fitness array of each iteration as parameter

    Return:
    np.ndarray: array of the best individuals
    np.ndarray: array of the best fitness
    int: duration of the whole method
    """
    best = []
    best_fitness = []
    start = t.time()
    for _ in progressbar.progressbar(range(nb_of_cycle)):
        fitness = total_fitness(population, time, temperature)
        round_best_index = np.where(fitness == min(fitness))[0][0]
        best.append(population[round_best_index])
        best_fitness.append(fitness[round_best_index])
        if stopFunction(best_fitness):
            break
        single_fit(population, time, temperature, number_of_parents, alpha_a_crossover, number_of_bc_flips, number_of_resets, number_of_a_variations, a_std_mutation, fitness, borne_a, borne_b, borne_c, a_precision)
    duration = t.time() - start
    fitness = total_fitness(population, time, temperature)
    round_best_index = np.where(fitness == min(fitness))[0][0]
    best.append(population[round_best_index])
    best_fitness.append(fitness[round_best_index])
    print("\nbest fitness : %s\nbest individual : %s\ndurations : %s" % (best_fitness[-1], best[-1], duration))
    return best, best_fitness, duration
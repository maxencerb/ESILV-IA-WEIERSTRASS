import numpy as np

def temperatureCalcul(i, population):
    """
    Calcul la température sur toute une populations

    Return:
    numpy.ndarray: liste 1D contenant toutes les températures de la population
    """
    size = len(population)
    sum, ai, bi = np.zeros(size), np.ones(size), np.ones(size)
    ipi = i * np.pi
    a, b, c = population[:,0], population[:,1], population[:,2]
    cint = int(max(c))
    test = np.ones(size, dtype=bool)
    for n in range(cint + 1):
        temp = ai * np.cos(ipi * bi)
        test = np.logical_and(test, n <= c)
        temp *= test.astype(int)
        sum += temp
        if n == cint:
            break
        ai = ai * a * test.astype(int)
        bi = bi * b * test.astype(int)
    return sum + np.random.normal(0, .01, size=len(sum))

def total_fitness(population, time, temperature):
    """
    Calcul la fitness sur toutes la population

    Return:
    numpy.ndarray: liste 1D contenant les fitness de la population
    """
    cost = np.zeros(len(population))
    for i in range(len(time)):
        cost += np.absolute(temperatureCalcul(time[i], population) - temperature[i])
    return cost / len(time)

def total_fitness2(population, time, temperature, abs_error):
    """
    Calcul la fitness sur toutes la population
    Test anti overfitting

    Return:
    numpy.ndarray: liste 1D contenant les fitness de la population
    """
    cost = np.zeros(len(population))
    for i in range(len(time)):
        iter_cost = np.absolute(temperatureCalcul(time[i], population) - temperature[i])
        index = iter_cost < abs_error
        iter_cost[index] = 0
        cost += iter_cost
    return cost / len(time)

def select_parents(population, fitness, number_of_parents = 20):
    """
    Selectionne des parents au hasard dans la population sans remise

    Return:
    np.ndarray: liste 3D contenant les parents (couples)
    """
    prob = 1/fitness
    prob = prob / sum(prob)
    choices = np.random.choice(population.shape[0], size=number_of_parents*2, replace=False, p=prob)
    parents = []
    for i in range(0, len(choices), 2):
        parents.append([population[choices[i]], population[choices[i + 1]]])
    parents = np.array(parents)
    return parents

def selection(population, childrens, fitness):
    """
    Remplace des individus au hasard proportionellement à leur fitness par les enfants
    """
    tokeep = np.where(fitness == min(fitness))[0][0]
    fitness = np.delete(fitness, tokeep)
    fitness = fitness/sum(fitness)
    change = np.random.choice(np.delete(np.arange(len(population)), tokeep), size=len(childrens), p=fitness, replace=False)
    for i in range(len(childrens)):
        population[change[i]] = childrens[i]
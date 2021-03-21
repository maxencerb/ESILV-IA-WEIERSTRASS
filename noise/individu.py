import math
import numpy as np
from init import set_variable_value, BORNE_A

def temperatureCalculIndividu(i, a, b, c):
    """
    Calcul la température donéée par l'individu à un instant i

    Return:
    float: la temprature
    """
    sum = .0
    ai = 1
    bi = 1
    ipi = i * math.pi
    cint = int(c)
    for n in range(cint + 1):
        sum += ai * math.cos(ipi * bi)
        if n == cint:
            break
        ai *= a
        bi *= b
    return sum

def fitnessCalculIndividu(a, b, c, time, temperature):
    """
    Calcul la fitness d'un individu sur le dataset en paramètre

    Return:
    float: moyenne pondéré des différences entre température estimé et température réél
    """
    cost = .0
    for i in range(len(time)):
        cost += abs(temperatureCalculIndividu(time[i], a, b, c) - temperature[i])
    return cost / len(time)

def total_fitness_individu(population, time, temperature):
    """
    Calcul la fitness sur toute la population

    Return:
    np.ndarray: liste 1D contenant les fitness de tous les individus
    """
    fitnessList = []
    for i in population:
        a, b, c = i
        fitnessList.append(fitnessCalculIndividu(a, b, c, time, temperature))
    return np.array(fitnessList)

def crossover(p1, p2, alpha_a_crossover = .3, borne_a = BORNE_A, a_precision = 2):
    """
    Croise les parents passé en paramètre pour créer deux autres enfants

    Return:
    np.ndarray: représente l'enfant 1
    np.ndarray: représente l'enfant 2
    """
    c1, c2 = np.zeros(3), np.zeros(3)
    c1[1], c2[1] = p1[1], p2[1]
    c1[2], c2[2] = p2[2], p1[2]
    c1[0], c2[0] = set_variable_value(p1[0] * alpha_a_crossover + p2[0] * (1 - alpha_a_crossover), borne_a, float, a_precision), set_variable_value(p2[0] * alpha_a_crossover + p1[0] * (1 - alpha_a_crossover), borne_a, float, a_precision)
    return c1, c2
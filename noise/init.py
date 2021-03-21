import numpy as np

EPS = .01 # np.finfo(float).eps
BORNE_A = [0+EPS, 1-EPS] 
BORNE_B = [1, 20] 
BORNE_C = [1, 20]


def readFile(filename, skipFirstLine = True):
    """
    Lit le fichier contenant les températures

    Return:
    np.ndarray: liste 1D contenant les temps
    np.ndarray: liste 1D contenant les températures
    """
    time, temperature = [], []
    file = open(filename, 'r')
    for line in file.readlines()[1 if skipFirstLine else 0:]:
        a, b = line.split(';')
        time.append(float(a))
        temperature.append(float(b))
    return np.array(time), np.array(temperature)

def set_variable_value(n, borne, type, a_precision = 2):
    """
    Met la valeur de la variable dans les bornes prédéfini et au format souhaité

    Return:
    type(type): valeur mise dans le bonne espace de recherche
    """
    n = float(n)
    if n < borne[0]:
        n = borne[0]
    elif n > borne[1]:
        n = borne[1]
    return round(type(n), a_precision)

def initPopulation(population_size = 100, borne_a = [0+EPS, 1-EPS], borne_b = [1, 20], borne_c = [1, 20]):
    """
    Random initialization

    Return:
    np.ndarray: Tableau 2D contenant toute la population
    """
    pop = []
    for _ in range(population_size):
        a = set_variable_value(np.random.random(), borne_a, float)
        b = np.random.randint(borne_b[0], high=borne_b[1])
        c = np.random.randint(borne_c[0], high=borne_c[1])
        pop.append([a, b, c])
    return np.array(pop)
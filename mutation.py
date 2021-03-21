import numpy as np
from init import set_variable_value

def b_c_flip_mutation(childrens, borne_b, borne_c, number_of_bc_flips = 5):
    index = np.random.choice(childrens.shape[0], size=number_of_bc_flips, replace=False)
    for i in index:
        childrens[i][1], childrens[i][2] = set_variable_value(childrens[i][2], borne_b, int), set_variable_value(childrens[i][1], borne_c, int)

def reset_mutation(childrens, borne_a, borne_b, borne_c, number_of_resets = 10, a_precision = 2):
    rows = np.random.choice(childrens.shape[0], size = number_of_resets, replace=False)
    columns = np.random.choice(childrens.shape[1], size = number_of_resets)
    for i in range(number_of_resets):
        if columns[i] == 0:
            childrens[rows[i]][columns[i]] = set_variable_value(np.random.random(), borne_a, float, a_precision)  
        elif columns[i] == 1:
             childrens[rows[i]][columns[i]] = np.random.randint(borne_b[0], high=borne_b[1])
        else:
            childrens[rows[i]][columns[i]] = np.random.randint(borne_c[0], high=borne_c[1])

def a_mutation(childrens, borne_a, number_of_a_variations = 10, a_std_mutation = .2, a_precision = 2):
    variations = np.random.normal(scale=a_std_mutation, size=number_of_a_variations)
    np.random.shuffle(childrens)
    for i in range(number_of_a_variations):
        childrens[i][0] = set_variable_value(childrens[i][0] + variations[i], borne_a, float, a_precision)
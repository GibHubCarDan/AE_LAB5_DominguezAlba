import pandas as pd
import numpy as np

# Dataset
data = {
    'ProjectID': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'],
    'Cost_Soles': [4932, 2191, 3074, 2613, 4646, 4047, 4286, 4183],
    'Benefit_Soles': [14535, 12995, 12862, 12176, 12598, 11985, 10524, 6827]
}
df = pd.DataFrame(data)

max_budget = 10000

# Función para calcular beneficio dado un bitstring


def fitness(bitstring):
    selected = [int(b) for b in bitstring]
    total_cost = np.dot(selected, df['Cost_Soles'])
    if total_cost > max_budget:
        return -np.inf
    total_benefit = np.dot(selected, df['Benefit_Soles'])
    return total_benefit


# Generar todas las combinaciones posibles (2^8 = 256)
n = len(df)
best_benefit = -np.inf
best_selection = None

for i in range(2**n):
    bitstring = format(i, '0{}b'.format(n))
    val = fitness(bitstring)
    if val > best_benefit:
        best_benefit = val
        best_selection = bitstring

# Mostrar resultados
selected_projects = [df['ProjectID'][i]
                     for i, bit in enumerate(best_selection) if bit == '1']

print("Proyectos seleccionados:", selected_projects)
print("Beneficio total:", best_benefit)

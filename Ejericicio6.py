import pandas as pd
import numpy as np

# Dataset
data = {
    'QuestionID': [f"Q{i}" for i in range(1, 31)],
    'Difficulty': [60, 40, 60, 70, 40, 40, 50, 40, 40, 60, 70, 80, 70, 70, 40, 80, 80, 70, 60, 80, 80, 70, 50, 80, 70, 50, 70, 70, 60, 40],
    'Time_min':   [20, 15, 15, 15, 15, 20, 10, 15, 20, 25, 15, 25, 15, 15, 20, 10, 20, 20, 25, 20, 20, 20, 10, 25, 15, 15, 25, 25, 20, 25]
}
df = pd.DataFrame(data).set_index('QuestionID')

# Parámetros
max_time = 90
min_difficulty = 180
max_difficulty = 200

n = len(df)

# Función de aptitud


def fitness(bitstring):
    # bitstring: np.array de 0/1 con longitud n
    selected = df[bitstring == 1]
    total_time = selected['Time_min'].sum()
    total_diff = selected['Difficulty'].sum()
    if total_time > max_time:
        return -np.inf
    if total_diff < min_difficulty or total_diff > max_difficulty:
        return -np.inf
    return total_diff  # Maximizar dificultad dentro del rango

# Generar vecino: voltear un bit


def get_neighbors(bitstring):
    neighbors = []
    for i in range(n):
        neighbor = bitstring.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors


# Inicialización aleatoria (puede no ser válida)
np.random.seed(42)
current = np.random.randint(0, 2, n)

# Mejorar solución con Hill Climbing
current_fit = fitness(current)
improved = True

while improved:
    improved = False
    neighbors = get_neighbors(current)
    for neigh in neighbors:
        fit = fitness(neigh)
        if fit > current_fit:
            current = neigh
            current_fit = fit
            improved = True
            break  # Primer vecino mejor

# Resultados
selected_questions = df.index[current == 1].tolist()
selected_time = df.loc[selected_questions, 'Time_min'].sum()
selected_diff = df.loc[selected_questions, 'Difficulty'].sum()

print("Preguntas seleccionadas:", selected_questions)
print(f"Tiempo total: {selected_time} minutos")
print(f"Dificultad total: {selected_diff}")
print("Bitstring:", current.tolist())

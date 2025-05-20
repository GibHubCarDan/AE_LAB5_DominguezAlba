from io import StringIO
import pandas as pd
import random

# Cargar datos
data = """MentorID Slot1 Slot2 Slot3 Slot4 Slot5 Slot6 Slot7 Slot8 Slot9 Slot10
M01 1 1 1 1 0 0 0 1 1 1
M02 1 1 0 1 0 0 1 0 0 1
M03 0 1 0 1 1 0 1 0 0 1
M04 0 1 0 0 0 0 1 1 0 1
M05 1 1 0 1 1 1 0 0 0 1
M06 1 0 0 0 1 1 0 1 0 1
M07 0 1 0 0 0 1 1 0 1 1
M08 1 1 1 1 1 1 0 1 1 1
M09 1 1 1 0 1 1 0 0 0 1
M10 1 0 1 1 1 0 1 0 0 0
M11 1 0 0 0 0 1 0 0 0 0
M12 1 1 0 0 0 0 0 0 0 1
M13 1 1 1 1 1 1 0 1 1 1
M14 0 0 1 0 1 0 1 1 1 1
M15 0 1 1 0 1 1 0 0 0 1
M16 1 1 0 0 1 0 1 1 0 0
M17 0 1 1 0 0 1 0 0 1 0
M18 1 0 0 0 1 1 1 1 0 1
M19 0 0 1 0 0 0 0 1 1 0
M20 0 0 0 0 1 0 1 0 1 0"""

df = pd.read_csv(StringIO(data), sep=" ")

# Generar todos los bloques válidos de 2 horas
block_list = [(i, i+1) for i in range(1, 10)]

# Verifica si un mentor puede cubrir un bloque de 2 horas


def is_available(row, block):
    return row[f"Slot{block[0]}"] == 1 and row[f"Slot{block[1]}"] == 1

# Generar una solución inicial válida


def generate_initial_solution(df):
    assignment = {}
    for _, row in df.iterrows():
        valid_blocks = [
            block for block in block_list if is_available(row, block)]
        if not valid_blocks:
            print(
                f"Mentor {row['MentorID']} excluido: sin bloque de 2h disponible.")
            continue  # Saltar mentor sin bloque válido
        assignment[row['MentorID']] = random.choice(valid_blocks)
    return assignment

# Calcular función de costo: cuántos mentores se asignaron al mismo bloque


def calculate_collisions(assignment):
    block_counts = {}
    for block in assignment.values():
        block_counts[block] = block_counts.get(block, 0) + 1
    return sum(count - 1 for count in block_counts.values() if count > 1)

# Generar vecindad: cambiar un bloque de un mentor


def generate_neighbors(assignment, df):
    neighbors = []
    for mentor in assignment:
        row = df[df['MentorID'] == mentor].iloc[0]
        current_block = assignment[mentor]
        for block in block_list:
            if block != current_block and is_available(row, block):
                new_assignment = assignment.copy()
                new_assignment[mentor] = block
                neighbors.append(new_assignment)
    return neighbors

# Algoritmo de búsqueda local


def hill_climbing(df):
    assignment = generate_initial_solution(df)
    current_cost = calculate_collisions(assignment)
    while True:
        neighbors = generate_neighbors(assignment, df)
        improved = False
        for neighbor in neighbors:
            cost = calculate_collisions(neighbor)
            if cost < current_cost:
                assignment = neighbor
                current_cost = cost
                improved = True
                break
        if not improved:
            break
    return assignment, current_cost


# Ejecutar el algoritmo
solution, collisions = hill_climbing(df)

# Mostrar resultados
print("\n--- Solución final ---")
for mentor, block in solution.items():
    print(f"{mentor}: Slot{block[0]} - Slot{block[1]}")
print(f"\nChoques: {collisions}")

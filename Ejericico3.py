from io import StringIO
import pandas as pd
import random

# Cargar la matriz de distancias (se reemplazan comas por puntos para floats)
data = """\tLab1\tLab2\tLab3\tLab4\tLab5\tLab6\tLab7\tLab8\tLab9\tLab10
Lab1\t0\t158\t140\t208\t80.5\t180\t183\t129.5\t111\t168.5
Lab2\t158\t0\t185.5\t146.5\t183.5\t170\t226\t197\t137.5\t177
Lab3\t140\t185.5\t0\t205.5\t111\t257\t150\t228.5\t167\t95
Lab4\t208\t146.5\t205.5\t0\t227\t223\t225\t157.5\t161.5\t238.5
Lab5\t80.5\t183.5\t111\t227\t0\t146\t243.5\t135.5\t211\t199
Lab6\t180\t170\t257\t223\t146\t0\t221\t161.5\t84\t173
Lab7\t183\t226\t150\t225\t243.5\t221\t0\t237\t184\t180.5
Lab8\t129.5\t197\t228.5\t157.5\t135.5\t161.5\t237\t0\t149.5\t169
Lab9\t111\t137.5\t167\t161.5\t211\t84\t184\t149.5\t0\t152
Lab10\t168.5\t177\t95\t238.5\t199\t173\t180.5\t169\t152\t0"""


df = pd.read_csv(StringIO(data), sep="\t", index_col=0)

# Función para calcular la distancia total de una ruta


def calculate_distance(route, df):
    total = 0
    for i in range(len(route) - 1):
        total += df.loc[route[i], route[i + 1]]
    total += df.loc[route[-1], route[0]]  # volver al inicio
    return total

# Generar vecinos por intercambio de dos nodos


def generate_neighbors(route):
    neighbors = []
    n = len(route)
    for i in range(n):
        for j in range(i + 1, n):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbors.append(new_route)
    return neighbors

# Algoritmo Hill Climbing


def hill_climbing(df, iterations=1000):
    labs = list(df.columns)
    current_route = random.sample(labs, len(labs))
    current_distance = calculate_distance(current_route, df)

    for _ in range(iterations):
        neighbors = generate_neighbors(current_route)
        improved = False
        for neighbor in neighbors:
            d = calculate_distance(neighbor, df)
            if d < current_distance:
                current_route = neighbor
                current_distance = d
                improved = True
                break
        if not improved:
            break

    return current_route, current_distance


# Ejecutar
best_route, best_distance = hill_climbing(df, 1000)

# Mostrar resultados
print("🔁 Ruta óptima encontrada:")
print(" → ".join(best_route + [best_route[0]]))
print(f"\n📏 Distancia total: {best_distance:.2f} metros")

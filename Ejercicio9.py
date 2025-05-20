import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('ejercicio9.csv', sep=';')

X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']].values
y = data['Spam'].values

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Configuración de DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.0, 1.0)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float, n=6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación


def evaluate(individual):
    weights = np.array(individual[:5])
    threshold = individual[5]
    scores = np.dot(X_val, weights)
    y_pred = (scores >= threshold).astype(int)
    return (f1_score(y_val, y_pred),)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Mutación corregida para mantener valores en [0, 1]


def bounded_mutate(individual, mu, sigma, indpb):
    tools.mutGaussian(individual, mu, sigma, indpb)
    for i in range(len(individual)):
        individual[i] = min(max(individual[i], 0.0), 1.0)
    return (individual,)


toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Hill Climbing


def hill_climb(individual, steps=10, step_size=0.05):
    best = individual[:]
    best_fitness = evaluate(best)[0]
    for _ in range(steps):
        neighbor = best[:]
        idx = random.randint(0, len(neighbor) - 1)
        neighbor[idx] += random.uniform(-step_size, step_size)
        neighbor[idx] = min(max(neighbor[idx], 0.0), 1.0)
        fitness = evaluate(neighbor)[0]
        if fitness > best_fitness:
            best = neighbor
            best_fitness = fitness
    ind = creator.Individual(best)
    ind.fitness.values = (best_fitness,)
    return ind

# Evolución


def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    fitnesses = []

    for gen in range(20):
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for i in range(len(offspring)):
            if not offspring[i].fitness.valid:
                offspring[i] = hill_climb(offspring[i])

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)
        fitnesses.append(record['max'])
        print(f"Generación {gen}: F1-score máximo = {record['max']}")

    best_ind = hof[0]
    print(f"\nMejor individuo: {best_ind}")
    print(f"F1-score en validación: {evaluate(best_ind)[0]}")

    plt.plot(fitnesses)
    plt.xlabel("Generación")
    plt.ylabel("F1-score máximo")
    plt.title("Evolución del F1-score")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

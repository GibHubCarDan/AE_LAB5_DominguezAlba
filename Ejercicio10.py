import random
import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------------
# Cargar y preparar el dataset
# -------------------------------
data = pd.read_csv('ejercicio10.csv', sep=';')

# Escalado de variables numéricas
scaler = StandardScaler()
X_num = scaler.fit_transform(
    data[['Credits', 'Prev_GPA', 'Extracurricular_hours']])

# Codificación de la variable objetivo
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(data['Category'])
y_cat = tf.keras.utils.to_categorical(y_int, num_classes=3)

# División en train y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_num, y_cat, test_size=0.3, random_state=RANDOM_SEED)

# -------------------------------
# Configuración DEAP para el algoritmo genético
# -------------------------------

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Rango para capas, neuronas y learning rate
toolbox.register("n_layers", random.randint, 1, 3)
toolbox.register("n_neurons", random.randint, 4, 64)
toolbox.register("learning_rate", random.uniform, 0.0001, 0.1)


def create_individual():
    n_layers = toolbox.n_layers()
    # Siempre 3 neuronas posibles, usamos n_layers para elegir cuántas
    neurons = [toolbox.n_neurons() for _ in range(3)]
    lr = toolbox.learning_rate()
    return creator.Individual([n_layers] + neurons + [lr])


toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def build_model(individual, input_shape):
    n_layers = individual[0]
    neurons = individual[1:1 + n_layers]
    learning_rate = individual[-1]

    model = Sequential()
    model.add(Dense(neurons[0], activation='relu', input_shape=(input_shape,)))
    for i in range(1, n_layers):
        model.add(Dense(neurons[i], activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate(individual, X_train, y_train, X_val, y_val):
    model = build_model(individual, X_train.shape[1])
    model.fit(X_train, y_train, epochs=15, verbose=0,
              batch_size=32, validation_data=(X_val, y_val))
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return (accuracy,)


toolbox.register("evaluate", evaluate, X_train=X_train,
                 y_train=y_train, X_val=X_val, y_val=y_val)
toolbox.register("mate", tools.cxUniform, indpb=0.5)


def custom_mutation(individual, mu=0, sigma=5, indpb=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:  # Número de capas
                individual[i] = random.randint(1, 3)
            elif i in [1, 2, 3]:  # Neuronas en capa
                individual[i] = max(
                    4, min(64, individual[i] + int(random.gauss(mu, sigma))))
            else:  # Learning rate
                individual[i] = min(
                    max(0.0001, individual[i] + random.gauss(0, 0.01)), 0.1)
    return (individual,)


toolbox.register("mutate", custom_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)


def hill_climb(individual, X_train, y_train, X_val, y_val, steps=5, step_size=1):
    best = individual[:]
    best_fitness = evaluate(best, X_train, y_train, X_val, y_val)[0]

    for _ in range(steps):
        neighbor = best[:]
        idx = random.randint(1, 3)  # Mutamos neuronas en capa
        neighbor[idx] = max(
            4, min(64, neighbor[idx] + random.choice([-step_size, step_size])))

        fitness = evaluate(neighbor, X_train, y_train, X_val, y_val)[0]
        if fitness > best_fitness:
            best = neighbor
            best_fitness = fitness

    return creator.Individual(best)


def main():
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    NGEN = 5

    for gen in range(NGEN):
        print(f"\nGeneración {gen + 1}")

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Cruce
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutación
        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Hill climbing local
        for i in range(len(offspring)):
            if not offspring[i].fitness.valid:
                offspring[i] = hill_climb(
                    offspring[i], X_train, y_train, X_val, y_val)

        # Evaluar población actualizada
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)
        print(record)

    best = hof[0]
    print("\n🧠 Mejor arquitectura encontrada:")
    print(f"- Capas ocultas: {best[0]}")
    print(f"- Neuronas por capa: {best[1:1 + best[0]]}")
    print(f"- Learning rate: {best[-1]:.5f}")
    print(
        f"- Accuracy final: {evaluate(best, X_train, y_train, X_val, y_val)[0]:.4f}")


if __name__ == "__main__":
    main()

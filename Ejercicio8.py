import random
import numpy as np
import pandas as pd
import io  # <-- Importar io para StringIO
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# --- Dataset (copiado y pegado en pandas) ---
data_text = """
Rooms,Area_m2,Price_Soles
3,105,6390
1,26,10940
4,68,10080
4,60,7560
1,46,9990
1,50,8050
2,25,9100
1,35,9140
2,83,6640
1,57,6660
2,79,6610
3,115,6940
4,116,10120
2,100,8490
4,52,10290
3,82,9710
2,30,10760
3,32,6610
2,113,4380
4,84,10750
3,49,4880
1,110,11190
3,42,7400
2,39,7700
4,25,6650
1,56,5320
1,103,4240
3,25,10060
3,112,7860
1,96,9890
3,46,6670
4,103,7970
1,32,8840
1,27,4810
3,72,5540
3,100,10870
2,49,10100
3,20,9330
1,107,10820
1,99,8410
2,94,5450
2,23,10130
1,81,6440
2,87,8690
1,36,5830
3,93,10120
1,104,5960
2,76,5110
3,82,6260
1,76,11790
1,68,4970
2,37,6380
2,77,8030
4,29,4110
1,82,9370
3,99,4970
2,98,7510
3,53,7010
4,24,11740
2,36,10010
3,91,8930
1,82,9500
3,43,4510
1,111,4160
3,27,11300
1,42,11710
1,118,8300
3,50,10340
2,100,8130
1,47,11700
4,67,10450
3,31,7770
4,107,11830
1,98,8190
1,50,5800
3,110,7590
4,54,6630
1,74,10910
4,101,7580
2,55,11610
3,91,10950
1,79,10250
3,57,7950
4,88,8510
3,73,11170
4,33,7110
4,39,10000
2,28,9070
3,116,6520
4,101,9040
3,89,5170
3,21,7710
3,49,8700
3,25,5850
1,100,7490
4,60,10140
3,42,11180
1,50,9940
1,39,6360
2,24,4030
2,40,5650
3,94,7910
2,87,11970
1,106,4780
4,80,11790
3,113,9230
2,61,11840
4,60,4600
1,80,10350
2,101,5030
3,71,11030
2,60,10990
3,79,5290
3,114,10930
2,49,6840
3,117,8480
3,114,11280
3,57,9850
4,108,7950
3,36,8760
2,119,7730
3,59,8390
3,49,4150
3,85,5310
4,43,5060
3,92,8720
1,20,6130
2,107,6840
2,57,11250
4,83,9900
1,46,6670
2,109,6810
1,109,7590
2,26,11790
4,108,11490
2,99,8180
4,61,10560
4,37,6570
3,100,9280
1,63,9450
4,89,4330
1,39,5720
2,42,8860
1,31,7280
3,31,6050
2,53,5060
2,27,9950
3,46,11540
1,68,8160
1,91,7240
"""

df = pd.read_csv(io.StringIO(data_text))  # CORRECCIÓN aquí

# --- Preparar datos ---
X = df[['Rooms', 'Area_m2']].values
y = df['Price_Soles'].values

# Dividir datos para evaluación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- DEAP setup ---

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar RMSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Individual: un solo float alpha en [0.0001, 10]
toolbox.register("attr_alpha", random.uniform, 0.0001, 10)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_alpha, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_ridge(individual):
    alpha = individual[0]
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return (rmse,)


toolbox.register("evaluate", eval_ridge)


# Mutación gaussiana pequeña (mu=0, sigma=0.1), con límite en [0.0001, 10]
def mut_gauss_limit(individual, mu=0, sigma=0.1, indpb=1.0):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            if individual[i] < 0.0001:
                individual[i] = 0.0001
            elif individual[i] > 10:
                individual[i] = 10
    return individual,


toolbox.register("mutate", mut_gauss_limit, mu=0, sigma=0.1, indpb=1.0)

# Sin cruce
toolbox.register("mate", lambda ind1, ind2: (ind1, ind2))

# Selección greedy (mejor vecino)


def sel_greedy(population, k):
    # Elige los mejores k individuos (mínimo RMSE)
    return tools.selBest(population, k)


toolbox.register("select", sel_greedy)


def main():
    random.seed(42)
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)  # Mejor individuo

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Algoritmo de hill climbing con población:
    # iteramos manualmente, mutamos y seleccionamos el mejor vecino
    max_gen = 50
    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg"]

    for gen in range(max_gen):
        # Mutar cada individuo
        offspring = []
        for ind in pop:
            mutant = toolbox.clone(ind)
            toolbox.mutate(mutant)
            del mutant.fitness.values
            offspring.append(mutant)

        # Evaluar nuevos individuos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Unir población original + mutantes
        combined = pop + offspring

        # Selección greedy de los mejores 20 individuos (población fija)
        pop = toolbox.select(combined, 20)

        # Actualizar Hall of Fame
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        print(
            f"Gen {gen}: min RMSE={record['min']:.4f}, avg RMSE={record['avg']:.4f}")

    best_alpha = hof[0][0]
    print(f"\nMejor alpha encontrado: {best_alpha:.4f}")

    # Graficar curva de convergencia
    gen = logbook.select("gen")
    min_rmse = logbook.select("min")
    avg_rmse = logbook.select("avg")

    plt.plot(gen, min_rmse, label="Min RMSE")
    plt.plot(gen, avg_rmse, label="Avg RMSE")
    plt.xlabel("Generación")
    plt.ylabel("RMSE")
    plt.title("Curva de convergencia - Optimización Ridge Alpha")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

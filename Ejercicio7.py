import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Dataset
data = {
    'StudentID': ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                  'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20'],
    'GPA': [14.24, 15.55, 12.87, 17.07, 14.15, 13.61, 18.29, 19.25, 10.46, 12.33,
            13.49, 18.15, 19.85, 19.69, 19.05, 12.97, 19.92, 12.49, 11.06, 19.51],
    'Skill': ['Backend', 'Data', 'Data', 'Data', 'Backend', 'Data', 'DevOps', 'DevOps', 'Backend', 'Frontend',
              'Data', 'Data', 'DevOps', 'Backend', 'Frontend', 'Data', 'Backend', 'Backend', 'Data', 'Frontend']
}
df = pd.DataFrame(data)

num_teams = 5
team_size = 4


def initial_solution():
    teams = [[] for _ in range(num_teams)]
    for i in range(len(df)):
        teams[i % num_teams].append(i)
    return teams


def skill_penalty(teams):
    penalty = 0
    all_skills = df['Skill'].unique()
    for skill in all_skills:
        counts = []
        for team in teams:
            count = sum(df.loc[member, 'Skill'] == skill for member in team)
            counts.append(count)
        penalty += np.var(counts)
    return penalty


def fitness(teams):
    gpa_vars = []
    for team in teams:
        gpas = df.loc[team, 'GPA']
        gpa_vars.append(np.var(gpas))
    total_variance = sum(gpa_vars)
    penalty = skill_penalty(teams)
    return total_variance + penalty


def get_neighbor(teams):
    new_teams = [team.copy() for team in teams]
    team_indices = np.random.choice(num_teams, 2, replace=False)
    t1, t2 = team_indices[0], team_indices[1]
    a1 = np.random.choice(new_teams[t1])
    a2 = np.random.choice(new_teams[t2])
    new_teams[t1].remove(a1)
    new_teams[t1].append(a2)
    new_teams[t2].remove(a2)
    new_teams[t2].append(a1)
    return new_teams


def local_search(max_iter=1000):
    current_solution = initial_solution()
    current_fitness = fitness(current_solution)
    for _ in range(max_iter):
        neighbor = get_neighbor(current_solution)
        neighbor_fitness = fitness(neighbor)
        if neighbor_fitness < current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
    return current_solution, current_fitness


best_teams, best_score = local_search(max_iter=5000)

# Mostrar resultados
print(f"Mejor función de aptitud: {best_score:.4f}\n")
for i, team in enumerate(best_teams):
    print(f"Equipo {i+1}:")
    team_students = df.loc[team]
    print(team_students[['StudentID', 'GPA', 'Skill']])
    print(f"  GPA varianza: {np.var(team_students['GPA']):.4f}")
    skill_counts = Counter(team_students['Skill'])
    print(f"  Habilidades: {dict(skill_counts)}\n")

print(f"Penalización total de habilidades: {skill_penalty(best_teams):.4f}")

# --- Gráfico de resultados ---

# Calcular promedios de GPA y conteos de habilidades por equipo
avg_gpa = []
skill_categories = df['Skill'].unique()
skill_counts_by_team = {skill: [] for skill in skill_categories}

for team in best_teams:
    team_data = df.loc[team]
    avg_gpa.append(team_data['GPA'].mean())
    counts = Counter(team_data['Skill'])
    for skill in skill_categories:
        skill_counts_by_team[skill].append(counts.get(skill, 0))

# Crear gráfico combinado
teams_labels = [f'Equipo {i+1}' for i in range(num_teams)]
x = np.arange(num_teams)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Barras del promedio GPA
bar_gpa = ax1.bar(x - 0.2, avg_gpa, width=0.4,
                  color='skyblue', label='Promedio GPA')
ax1.set_ylabel('Promedio GPA')
ax1.set_ylim(0, 22)
ax1.set_xticks(x)
ax1.set_xticklabels(teams_labels)
ax1.set_title(
    'Balanceo de Equipos: Promedio GPA y Distribución de Habilidades')

# Segundo eje para habilidades
ax2 = ax1.twinx()
bottoms = np.zeros(num_teams)
colors = ['#ff9999', '#66b3ff', '#99ff99',
          '#ffcc99', '#c2c2f0']  # colores para skills

for i, skill in enumerate(skill_categories):
    counts = skill_counts_by_team[skill]
    ax2.bar(x + 0.2, counts, width=0.4, bottom=bottoms,
            color=colors[i % len(colors)], label=skill)
    bottoms += counts

ax2.set_ylabel('Cantidad de alumnos por habilidad')
ax2.set_ylim(0, team_size+1)

# Leyenda combinada
bars_labels = [bar_gpa]
bars_labels += [plt.Rectangle((0, 0), 1, 1, color=colors[i])
                for i in range(len(skill_categories))]
labels = ['Promedio GPA'] + list(skill_categories)
ax1.legend(bars_labels, labels, loc='upper left')

plt.tight_layout()
plt.show()

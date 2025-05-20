import pandas as pd
import random

# Paso 1: Cargar los datos
from io import StringIO

csv_data = """
StudentID,Parcial1,Parcial2,Parcial3
A001,17,10,18
A002,10,19,12
A003,5,5,13
A004,8,6,13
A005,16,7,14
A006,8,9,7
A007,12,7,13
A008,14,5,11
A009,8,18,16
A010,10,8,11
A011,7,7,18
A012,9,15,14
A013,12,18,6
A014,11,5,11
A015,13,12,17
A016,13,10,13
A017,17,14,18
A018,15,5,13
A019,6,15,8
A020,11,7,7
A021,12,16,8
A022,12,15,15
A023,19,12,17
A024,13,16,11
A025,6,7,8
A026,10,14,11
A027,14,7,19
A028,18,19,18
A029,13,8,10
A030,14,19,12
A031,9,16,16
A032,8,8,5
A033,5,7,16
A034,8,19,13
A035,10,8,9
A036,19,9,15
A037,5,6,11
A038,7,7,16
A039,8,16,10
A040,13,19,18
A041,6,14,17
A042,8,15,13
A043,18,6,7
A044,8,9,16
A045,8,15,8
A046,19,11,14
A047,12,16,12
A048,5,13,10
A049,6,16,8
A050,14,7,19
A051,14,8,9
A052,5,5,10
A053,15,5,18
A054,9,11,8
A055,12,5,8
A056,8,11,12
A057,19,8,14
A058,16,19,14
A059,7,15,14
A060,12,8,12
A061,17,13,8
A062,7,17,19
A063,5,13,15
A064,5,18,7
A065,9,19,17
A066,10,13,8
A067,10,18,14
A068,11,7,15
A069,13,8,16
A070,9,19,18
A071,6,7,12
A072,9,16,12
A073,14,18,18
A074,15,5,10
A075,15,13,6
A076,13,13,17
A077,6,8,7
A078,6,13,7
A079,12,15,13
A080,14,7,6
A081,14,13,10
A082,8,9,13
A083,11,8,9
A084,12,17,5
A085,16,18,16
A086,19,19,7
A087,7,5,10
A088,16,9,19
A089,5,8,10
A090,19,18,19
A091,8,16,5
A092,10,17,19
A093,17,11,13
A094,14,18,18
A095,15,14,6
A096,9,18,6
A097,16,16,5
A098,9,13,8
A099,11,5,13
A100,9,13,16
A101,9,18,13
A102,8,10,9
A103,17,19,9
A104,9,14,5
A105,9,5,14
A106,13,17,8
A107,19,14,12
A108,9,11,17
A109,8,10,8
A110,15,18,7
A111,12,8,19
A112,18,6,18
A113,10,13,6
A114,10,5,6
A115,5,9,7
A116,6,16,6
A117,10,16,18
A118,14,14,9
A119,8,11,18
A120,5,10,7
"""

data = pd.read_csv(StringIO(csv_data.strip()))

# Paso 2: Función de aptitud


def fitness(offset, df):
    adjusted = df[["Parcial1", "Parcial2", "Parcial3"]] + offset
    adjusted = adjusted.clip(upper=20)  # Limitar a máximo 20
    averages = adjusted.mean(axis=1)
    aprobados = (averages >= 11).sum()
    promedio_clase = averages.mean()

    # Penalizar si promedio general > 14
    if promedio_clase > 14:
        return aprobados - (promedio_clase - 14) * 10  # Penalización fuerte
    return aprobados


# Paso 3: Hill Climbing
current_offset = 0.0
best_offset = current_offset
best_score = fitness(current_offset, data)

for _ in range(1000):  # intentos
    new_offset = current_offset + random.choice([-0.5, 0.5])
    if -5 <= new_offset <= 5:
        new_score = fitness(new_offset, data)
        if new_score > best_score:
            best_score = new_score
            best_offset = new_offset
            current_offset = new_offset

# Paso 4: Resultado final
final_adjusted = data[["Parcial1", "Parcial2", "Parcial3"]] + best_offset
final_adjusted = final_adjusted.clip(upper=20)
data["PromedioFinal"] = final_adjusted.mean(axis=1)
data["Aprobado"] = data["PromedioFinal"] >= 11

print(f"\nOffset óptimo encontrado: {best_offset}")
print(f"Porcentaje de aprobados: {data['Aprobado'].mean() * 100:.2f}%")
print(f"Promedio general: {data['PromedioFinal'].mean():.2f}")

# Mostrar primeras filas
print("\nNueva distribución (primeros 10 alumnos):")
print(data[["StudentID", "PromedioFinal", "Aprobado"]].head(10))

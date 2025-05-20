import pandas as pd

data = {
    'TesistaID': ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15'],
    'F1': [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    'F2': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    'F3': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    'F4': [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    'F5': [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    'F6': [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data).set_index('TesistaID')

tesistas = df.index.tolist()
franjas = df.columns.tolist()
num_salas = 6
max_continuous = 4

# Calendario: sala -> franja -> tesista (solo uno)
calendario = {sala: {fr: None for fr in franjas}
              for sala in range(1, num_salas+1)}


def puede_asignar(sala, franja_idx):
    # Contamos franjas ocupadas, incluyendo la que se quiere asignar
    franjas_ocupadas = []
    for i, fr in enumerate(franjas):
        if calendario[sala][fr] is not None or i == franja_idx:
            franjas_ocupadas.append(i)
    franjas_ocupadas = sorted(set(franjas_ocupadas))
    max_cont = 1
    cont = 1
    for i in range(1, len(franjas_ocupadas)):
        if franjas_ocupadas[i] == franjas_ocupadas[i-1] + 1:
            cont += 1
            max_cont = max(max_cont, cont)
        else:
            cont = 1
    return max_cont <= max_continuous


asignaciones = {}

for t in tesistas:
    asignado = False
    for fr_idx, fr in enumerate(franjas):
        if df.loc[t, fr] == 1:
            for sala in range(1, num_salas+1):
                if calendario[sala][fr] is None and puede_asignar(sala, fr_idx):
                    calendario[sala][fr] = t
                    asignaciones[t] = (sala, fr)
                    asignado = True
                    break
            if asignado:
                break
    if not asignado:
        asignaciones[t] = (None, None)

# Métrica huecos por sala


def contar_huecos(sala_cal):
    ocupadas = [1 if sala_cal[fr] is not None else 0 for fr in franjas]
    if 1 not in ocupadas:
        return 0
    first = ocupadas.index(1)
    last = len(ocupadas) - 1 - ocupadas[::-1].index(1)
    huecos = ocupadas[first:last+1].count(0)
    return huecos


huecos_totales = sum(contar_huecos(calendario[s]) for s in calendario)

# Mostrar resultados
print("Asignación (Tesista -> Sala, Franja):")
for t in asignaciones:
    print(f"{t}: {asignaciones[t]}")

print("\nHuecos totales:", huecos_totales)

cal_df = pd.DataFrame(index=franjas, columns=[
                      f'Sala {s}' for s in range(1, num_salas+1)])
for sala in calendario:
    for fr in franjas:
        val = calendario[sala][fr]
        cal_df.loc[fr, f'Sala {sala}'] = val if val is not None else '-'

print("\nCalendario final:")
print(cal_df)

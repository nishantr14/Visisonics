from cyber_layer import get_cyber_score

sample = {
  'F_PU1': 1.2, 'F_PU2': 0.8,
  'P_J280': 55.0, 'P_J269': 48.0,
  'L_T1': 1.2, 'L_T2': 3.5,
  'L_T3': 2.1, 'S_PU1': 1
}

print(get_cyber_score(sample))
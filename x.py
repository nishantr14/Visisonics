import sys
import os
sys.path.append(os.path.abspath("."))

from src.physical_layer import get_physical_score

sample = {
    'L_T1': 1.2, 'L_T2': 3.5, 'L_T3': 2.1,
    'P_J280': 55.0, 'P_J269': 48.0,
    'F_PU1': 1.2, 'F_PU2': 0.8
}

print(get_physical_score(sample))
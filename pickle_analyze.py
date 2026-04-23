import pickle
import numpy as np

with open('experiments/thesis_experiments/results_220426_epsdiff_ei/gibo_ei_epsdiff/dim_8/run_001.pkl', 'rb') as f:
    d = pickle.load(f)
    
print(d.keys())

# Was drin ist:
print(d['step_sizes'][:10])          # EI alpha* pro Outer-Iteration
#print(d['inner_loop_samples'][:10])  # Samples pro Inner Loop
#print(d['regret_per_eval'])          # Regret nach jeder Eval
print(d['wolfe_satisfied'][:10])     # True/False pro Iteration
#print(d['f_max'])                    # Optimum der Funktion
#print(d['seed'], d['dimension'])     # Metadaten
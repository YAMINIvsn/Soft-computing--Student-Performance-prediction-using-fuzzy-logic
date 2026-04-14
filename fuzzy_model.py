import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_fuzzy_system():

    study = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'study')
    absences = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'absences')
    g1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'g1')
    g2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'g2')

    g3 = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'g3')

    # Membership functions
    study['low'] = fuzz.trimf(study.universe, [0, 0, 0.5])
    study['medium'] = fuzz.trimf(study.universe, [0.2, 0.5, 0.8])
    study['high'] = fuzz.trimf(study.universe, [0.5, 1, 1])

    absences['low'] = fuzz.trimf(absences.universe, [0, 0, 0.4])
    absences['medium'] = fuzz.trimf(absences.universe, [0.2, 0.5, 0.8])
    absences['high'] = fuzz.trimf(absences.universe, [0.6, 1, 1])

    for var in [g1, g2]:
        var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
        var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
        var['high'] = fuzz.trimf(var.universe, [0.5, 1, 1])

    g3['low'] = fuzz.trimf(g3.universe, [0, 0, 0.5])
    g3['medium'] = fuzz.trimf(g3.universe, [0.3, 0.5, 0.7])
    g3['high'] = fuzz.trimf(g3.universe, [0.5, 1, 1])

    rules = [

        ctrl.Rule(g1['high'] & g2['high'], g3['high']),
        ctrl.Rule(g1['medium'] & g2['high'], g3['high']),
        ctrl.Rule(g1['high'] & g2['medium'], g3['high']),

        ctrl.Rule(g1['medium'] & g2['medium'], g3['medium']),
        ctrl.Rule(g1['low'] & g2['low'], g3['low']),

        ctrl.Rule(g1['low'] & g2['high'], g3['medium']),
        ctrl.Rule(g1['high'] & g2['low'], g3['medium']),

        ctrl.Rule(study['high'] & g2['high'], g3['high']),
        ctrl.Rule(study['medium'] & g2['medium'], g3['medium']),
        ctrl.Rule(study['low'] & g2['low'], g3['low']),

        ctrl.Rule(absences['high'], g3['low']),
        ctrl.Rule(absences['low'] & study['high'], g3['high']),
        ctrl.Rule(absences['medium'] & g2['medium'], g3['medium']),

        ctrl.Rule(study['high'] & absences['low'] & g2['high'], g3['high']),
        ctrl.Rule(study['low'] & absences['high'], g3['low']),

        ctrl.Rule(g1['medium'] & absences['low'], g3['medium']),
        ctrl.Rule(g2['medium'] & study['medium'], g3['medium']),

        ctrl.Rule(g1['high'] & absences['low'], g3['high']),
        ctrl.Rule(g2['high'] & absences['low'], g3['high']),

        ctrl.Rule(g1['low'] & absences['high'], g3['low']),
        ctrl.Rule(g2['low'] & absences['high'], g3['low']),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)
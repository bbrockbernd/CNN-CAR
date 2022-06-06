import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from CustomOptimizer import CustomOptimizer
from DataLoader import DataLoader, DataSet
import keras_tuner as kt

from Model1D import HyperModel1D

TOP_TRIALS = 20


tuner = CustomOptimizer(HyperModel1D(),
                        objective="val_accuracy",
                        max_trials=100,
                        overwrite=False,
                        directory="supercompute/tuning_1D2_reading",
                        project_name="tune_hypermodel")

scores = ["{:.3f}".format(x.score) for x in tuner.oracle.get_best_trials(TOP_TRIALS)]

print(f'score : {scores}')
for key in tuner.oracle.hyperparameters._hps.keys():
    results= []
    for trial in tuner.oracle.get_best_trials(TOP_TRIALS):
        if key in trial.hyperparameters.values.keys():
            results.append(trial.hyperparameters.values[key])
        else:
            results.append(None)

    print(f'{key} : {results}')

# tuner.get_best_models(3)[0]
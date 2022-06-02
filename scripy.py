import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from CustomOptimizer import CustomOptimizer
from DataLoader import DataLoader, DataSet
import keras_tuner as kt

from Model1D import HyperModel1D

tuner = CustomOptimizer(HyperModel1D(),
                        objective="val_accuracy",
                        max_trials=100,
                        overwrite=False,
                        directory="tuning_tester",
                        project_name="tune_hypermodel")



for key in tuner.get_best_hyperparameters(10)[0].values.keys():
    results= []
    for hp in tuner.get_best_hyperparameters(10):
        results.append(hp.values[key])

    print(f'{key} : {results}')


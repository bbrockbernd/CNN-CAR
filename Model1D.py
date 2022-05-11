import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from CustomOptimizer import CustomOptimizer
from DataLoader import DataLoader, DataSet
import keras_tuner as kt


class HyperModel1D(kt.HyperModel):

    def build(self, hp):
        model = keras.Sequential()

        n_filters = hp.Int('filter count', 8, 128, 8)
        kernel_size = hp.Int('kernel size', 3, 17, 2)
        frame_length = hp.Int('frame length', 32, 512, 32)
        pool_1 = hp.Choice('max pool 1', [2, 4, 8, 16])
        dense_size = hp.Int('Dense size', 60, 200, 20)

        drop_1 = hp.Float('drop out 1', 0.0, 0.8, 0.2)
        drop_2 = hp.Float('drop out 2', 0.0, 0.8, 0.2)

        model.add(layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu', input_shape=(frame_length, 2), padding='same'))
        model.add(layers.Dropout(drop_1))
        model.add(layers.MaxPooling1D(pool_size=pool_1))
        model.add(layers.Flatten())
        model.add(layers.Dense(dense_size, activation='relu'))
        model.add(layers.Dropout(drop_2))
        model.add(layers.Dense(8, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        loader = DataLoader(DataSet.SEDENTARY)
        frame_length = hp.get('frame length')
        trainX, trainy, testX, testy = loader.load_1D(framelength=frame_length)

        return model.fit(trainX, trainy, *args, batch_size=64, validation_data=(testX, testy), **kwargs)



def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 20, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling1D(pool_size=4))
    # model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(testX, testy))
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

def tune_params():
    # tuner = kt.Hyperband(HyperModel1D(),
    #                      objective='val_accuracy',
    #                      max_epochs=30,
    #                      factor=3,
    #                      directory='tuning1d',
    #                      project_name='tune_hypermodel',
    #                      hyperband_iterations=2,
    #                      overwrite=True)
    # tuner.search()


    tuner = CustomOptimizer(HyperModel1D(),
                            objective="val_accuracy",
                            max_trials=100,
                            overwrite=True,
                            directory="tuning_tester",
                            project_name="tune_hypermodel")
    tuner.search(epochs=1)


# loader = DataLoader(DataSet.SEDENTARY)
# trainX, trainy, testX, testy = loader.load_1D()
# evaluate_model(trainX, trainy, testX, testy)

tune_params()
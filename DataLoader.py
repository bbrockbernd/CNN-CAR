from enum import Enum

from scipy.io import loadmat
import numpy as np
import os
import cv2
# from matplotlib import pyplot as plt


class DataSet(Enum):
    DESKTOP = 0
    READING = 1
    SEDENTARY = 2

class DataLoader:

    def __init__(self, ds: DataSet):
        self.dataPath: str = ['DesktopActivity', 'ReadingActivity', 'SedentaryActivity/data/subject'][ds.value]

    def load_1D(self, validation = 0.1, framelength = 256, per_frame_norm = False):
        subject_files = np.array(os.listdir(self.dataPath))
        n_test_subjects = int(np.ceil(validation * len(subject_files)))
        test_mask = np.zeros_like(subject_files, dtype=bool)
        test_mask[np.random.choice(len(subject_files), size=n_test_subjects, replace=False)] = True
        train_mask = ~test_mask

        trainX, trainy = None, None
        for file in subject_files[train_mask]:
            x, y = self.load_file(f'{self.dataPath}/{file}', framelength, per_frame_norm)
            if trainX is None:
                trainX = x
                trainy = y
            else:
                trainX = np.concatenate((x, trainX))
                trainy = np.concatenate((y, trainy))

        testX, testy = None, None
        for file in subject_files[test_mask]:
            x, y = self.load_file(f'{self.dataPath}/{file}', framelength, per_frame_norm)
            if testX is None:
                testX = x
                testy = y
            else:
                testX = np.concatenate((x, testX))
                testy = np.concatenate((y, testy))

        if not per_frame_norm:
            min = np.min([np.min(trainX), np.min(testX)])
            trainX, testX = trainX - min, testX - min

            max = np.max([np.max(trainX), np.max(testX)])
            trainX, testX = trainX / max, testX / max

        return trainX, trainy, testX, testy

    def load_file(self, file_name, framelength, per_frame_norm):
        dic = loadmat(file_name)
        keys = []
        for key in dic.keys():
            if key[0] != '_':
                keys.append(key)

        x = np.zeros((0, framelength, 2))
        y = np.zeros((0, len(keys)))

        for i, key in enumerate(keys):
            activity_data = dic[key]

            indices = np.array([np.arange(0, len(activity_data), framelength/2)[:-2], np.arange(framelength, len(activity_data), framelength/2)]).astype(int)
            frames = []
            for ind in indices.T:
                frame = activity_data[ind[0] : ind[1], 0:2]
                if per_frame_norm:
                    frame = frame - np.min(frame)
                    frame = frame / np.max(frame)
                frames.append(frame)
            framesX = np.array(frames)

            # Making y
            classy = np.zeros(len(keys))
            classy[i] = 1
            framesy = np.tile(classy, (framesX.shape[0], 1))

            x = np.concatenate((x, framesX))
            y = np.concatenate((y, framesy))

        return x, y

    def transform_to_2d(self, data, resolution=500, thickness=1):
        data2d = np.zeros((data.shape[0], resolution, resolution, 3), dtype=np.uint8)
        for i in range(data.shape[0]):
            current = data[i]
            current = np.floor(current * resolution).astype(int)
            current[current == resolution] = resolution - 1
            cv2.polylines(data2d[i], np.int32([current]), False, color=(255, 255, 255), thickness=thickness)

        data2dbw = np.zeros((data2d.shape[0], data2d.shape[1], data2d.shape[2]))
        data2dbw[data2d[:, :, :, 0] == 255] = 1
        return data2dbw[:, :, :, np.newaxis]



# trainX, trainy, testX, testy = DataLoader(DataSet.SEDENTARY).load_1D(per_frame_norm=True)
# trainX2D = DataLoader(DataSet.SEDENTARY).transform_to_2d(trainX, 200)
#
# plt.imshow(trainX2D[0])
# plt.show()

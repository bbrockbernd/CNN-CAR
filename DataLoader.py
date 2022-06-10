import time
from enum import Enum

# from matplotlib import pyplot as plt
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

    def load_1D(self, validation = 0.1, framelength = 256, testsubjects=[]):
        subject_files = np.array(os.listdir(self.dataPath))
        test_mask = np.zeros_like(subject_files, dtype=bool)
        if len(testsubjects) == 0:
            n_test_subjects = int(np.ceil(validation * len(subject_files)))
            test_mask[np.random.choice(len(subject_files), size=n_test_subjects, replace=False)] = True
        else:
            test_mask[testsubjects] = True
        print(f'Datasplit: {test_mask}')
        train_mask = ~test_mask

        trainX, trainy = None, None
        for file in subject_files[train_mask]:
            x, y = self.load_file(f'{self.dataPath}/{file}', framelength)
            if trainX is None:
                trainX = x
                trainy = y
            else:
                trainX = np.concatenate((x, trainX))
                trainy = np.concatenate((y, trainy))

        testX, testy = None, None
        for file in subject_files[test_mask]:
            x, y = self.load_file(f'{self.dataPath}/{file}', framelength)
            if testX is None:
                testX = x
                testy = y
            else:
                testX = np.concatenate((x, testX))
                testy = np.concatenate((y, testy))

        # min = np.min([np.min(trainX), np.min(testX)])
        # trainX, testX = trainX - min, testX - min
        #
        # max = np.max([np.max(trainX), np.max(testX)])
        # trainX, testX = trainX / max, testX / max

        return trainX, trainy, testX, testy

    def load_file(self, file_name, framelength):
        dic = loadmat(file_name)
        keys = []
        for key in dic.keys():
            if key[0] != '_':
                keys.append(key)

        x = np.zeros((0, framelength, 2))
        y = np.zeros((0, len(keys)))

        for i, key in enumerate(keys):
            activity_data = dic[key]

            # Remove outliers in X data[abs(data - np.mean(data)) < m * np.std(data)]
            activity_data = activity_data[np.abs(activity_data[:, 0] - np.mean(activity_data[:, 0])) < 2 * np.std(activity_data[:, 0])]
            activity_data = activity_data[np.abs(activity_data[:, 1] - np.mean(activity_data[:, 1])) < 2 * np.std(activity_data[:, 1])]

            # Sliding window
            indices = np.tile(np.arange(framelength), (len(activity_data) - framelength, 1)) + np.arange(len(activity_data) - framelength)[:, np.newaxis]
            framesX = activity_data[indices, 0:2]

            # Reduce data size
            framesX = framesX[::30]

            # Normalize
            framesX = framesX - np.min(framesX)
            framesX = framesX / np.max(framesX)

            # Making y
            classy = np.zeros(len(keys))
            classy[i] = 1
            framesy = np.tile(classy, (framesX.shape[0], 1))

            x = np.concatenate((x, framesX))
            y = np.concatenate((y, framesy))

        return x, y


    def points_to_gradient_image(self, data, resolution):
        data2d = np.zeros((data.shape[0], resolution, resolution), dtype=float)
        for i in range(data.shape[0]):
            #select current window
            current = data[i]

            #scale normalized coordinates with resolution
            current = np.floor(current * resolution).astype(int)
            current[current == resolution] = resolution - 1

            #Create line points array containing start and end point for each line to draw
            line_points = np.array([current, np.roll(current, -1, axis=0)]).swapaxes(0, 1)[:-1]

            #Create masks for horizontal and vertical lines
            horizontal_mask = np.abs(line_points[:, 1, 1] - line_points[:, 0, 1]) < np.abs(line_points[:, 1, 0] - line_points[:, 0, 0])
            vertical_mask = ~horizontal_mask
            vertical_mask[np.abs(line_points[:, 1, 1] - line_points[:, 0, 1]) == 0] = False

            #Initialize line coordinates by ranging resolution
            line_coords = np.tile(np.arange(resolution), (len(horizontal_mask), 2, 1))

            #set out of bounds to bounds
            lbx, hbx = np.min(line_points[:, :, 0], axis=1), np.max(line_points[:, :, 0], axis=1)
            lby, hby = np.min(line_points[:, :, 1], axis=1), np.max(line_points[:, :, 1], axis=1)

            line_coords[:, 0, :][line_coords[:, 0, :] < np.tile(lbx, (resolution, 1)).T] = np.tile(lbx, (resolution, 1)).T[line_coords[:, 0, :] < np.tile(lbx, (resolution, 1)).T]
            line_coords[:, 0, :][line_coords[:, 0, :] > np.tile(hbx, (resolution, 1)).T] = np.tile(hbx, (resolution, 1)).T[line_coords[:, 0, :] > np.tile(hbx, (resolution, 1)).T]

            line_coords[:, 1, :][line_coords[:, 1, :] < np.tile(lby, (resolution, 1)).T] = np.tile(lby, (resolution, 1)).T[line_coords[:, 1, :] < np.tile(lby, (resolution, 1)).T]
            line_coords[:, 1, :][line_coords[:, 1, :] > np.tile(hby, (resolution, 1)).T] = np.tile(hby, (resolution, 1)).T[line_coords[:, 1, :] > np.tile(hby, (resolution, 1)).T]

            #calculate slopes for hor and ver
            hor_slopes = (line_points[horizontal_mask, 1, 1] - line_points[horizontal_mask, 0, 1]) / (line_points[horizontal_mask, 1, 0] - line_points[horizontal_mask, 0, 0])
            y_offset =  line_points[horizontal_mask, 0, 1] - (line_points[horizontal_mask, 0, 0] * hor_slopes)

            ver_slopes = (line_points[vertical_mask, 1, 0] - line_points[vertical_mask, 0, 0]) / (line_points[vertical_mask, 1, 1] - line_points[vertical_mask, 0, 1])
            x_offset = line_points[vertical_mask, 0, 0] - (line_points[vertical_mask, 0, 1] * ver_slopes)

            #Create line coords
            line_coords[horizontal_mask, 1] = np.round(line_coords[horizontal_mask, 0] * hor_slopes[:, np.newaxis] + y_offset[:, np.newaxis])
            line_coords[vertical_mask, 0] = np.round(line_coords[vertical_mask, 1] * ver_slopes[:, np.newaxis] + x_offset[:, np.newaxis])

            #flatten coords and create gradients
            xs = line_coords[:, 0, :].flatten()
            ys = line_coords[:, 1, :].flatten()
            gradient = np.tile(np.linspace(0, 1, len(horizontal_mask)), (resolution, 1)).T.flatten()

            #set draw lines on image with gradient
            image = np.zeros((resolution, resolution))
            image[(xs, ys)] = gradient

            data2d[i] = image
        return data2d[:, :, :, np.newaxis]


    def transform_to_2d(self, data, resolution=500, thickness=1, gradient=True):
        # return self.points_to_gradient_image(data, resolution)

        if not gradient:
            data2d = np.zeros((data.shape[0], resolution, resolution, 3), dtype=float)
            for i in range(data.shape[0]):
                # colors = np.linspace(0, 1, data.shape[1])
                current = data[i]
                current = np.floor(current * resolution).astype(int)
                current[current == resolution] = resolution - 1
                cv2.polylines(data2d[i], np.int32([current]), False, color=(255, 255, 255), thickness=thickness)
                # print("image")

            data2dbw = np.zeros((data2d.shape[0], data2d.shape[1], data2d.shape[2]))
            data2dbw[data2d[:, :, :, 0] == 255] = 1
            return data2dbw[:, :, :, np.newaxis]

        else:
            data2d = np.zeros((data.shape[0], resolution, resolution, 3), dtype=float)
            for i in range(data.shape[0]):
                colors = np.linspace(0, 1, data.shape[1] - 1)
                current = data[i]
                current = np.floor(current * resolution).astype(int)
                current[current == resolution] = resolution - 1
                for j in range(current.shape[0] - 1):
                    cv2.line(data2d[i], (current[j, 0], current[j, 1]), (current[j + 1, 0], current[j + 1, 1]), color=(colors[j], colors[j], colors[j]), thickness=thickness)
            data2dbw = data2d[:, :, :, 0]
            return data2dbw[:, :, :, np.newaxis]

# loader = DataLoader(DataSet.DESKTOP)
# trainX, trainy, testX, testy = loader.load_1D(framelength=1024)
# s = time.time()
# joe = loader.transform_to_2d(testX, resolution=256)
# e = time.time()
# print(f'Time {e - s}')
import h5py
import os
import pickle
from matplotlib import pyplot as plt
from tensorflow import keras
import h5py
import numpy as np


def printConfusionMatrix(kerasModel, datagroup, dataset):
    predictions = kerasModel.predict(dataset[0])  # feed the numpy arrays to predict

    confMatrix = np.zeros([2, 2])  # rows are actual values
    for i in range(len(predictions)):  # cols are predictions
        pred = predictions[i]
        lbl = dataset[1][i]
#        print(lbl, pred)
        if lbl == 0:  # a row of true backgrounds
            if pred < 0.5:
                confMatrix[0, 0] += 1  # True negative
            else:
                confMatrix[0, 1] += 1  # False negative
        if lbl == 1:  # a row of true signals
            if pred < 0.5:
                confMatrix[1, 0] += 1  # False positive
            else:
                confMatrix[1, 1] += 1  # True positive

    print("Confusion matrix fractions for predictions on dataset", datagroup, "of length", len(dataset[0]))
    confMatrix = confMatrix / np.sum(confMatrix)  # normalize to fractions
    print('             PREDICT   0            1')
    print('   Label = 0      TN {:0.3f}    FN {:0.3f}'.format(confMatrix[0,0], confMatrix[0,1]))
    print('   Label = 1      FP {:0.3f}    TP {:0.3f}'.format(confMatrix[1,0], confMatrix[1,1]))


#################################################################
baseDir = "/home/val/PycharmProjects/github/signal-classifier/modelOutput/"


classifierModelFilename = "models/Classifier_0_10_epochs"
classifier_Model = keras.models.load_model(classifierModelFilename, compile = False)

h5filename = "h5fileHWcalls_1110records.h5"
#h5filename = "/home/val/PycharmProjects/github/signal-annotation/h5fakeSpecs.h5"
h5db = h5py.File(h5filename, mode='r')

train_specs = h5db['train_specs']
train_labels = h5db['train_labels']
test_specs = h5db['test_specs']
test_labels = h5db['test_labels']
eval_specs = h5db['eval_specs']
eval_labels = h5db['eval_labels']
#
# print("num of eval_specs records is", eval_specs.shape[0])

print("Classifier Model:")
print(classifier_Model.summary())

printConfusionMatrix(classifier_Model, "train dataset", (train_specs, train_labels))
printConfusionMatrix(classifier_Model, "test dataset", (test_specs, test_labels))
printConfusionMatrix(classifier_Model, "eval dataset", (eval_specs, eval_labels))


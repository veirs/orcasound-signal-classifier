import time
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import ReLU, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from keras.layers import LeakyReLU, ActivityRegularization
from keras.layers import Flatten, Dense, ReLU, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import h5py
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
#######################################

def encoderAE(inputs, layers):
    """ Construct the Encoder
        inputs : the input vector
        layers : number of filters per layer    """
    x = inputs
    # Feature pooling by 1/2H x 1/2W
    for n_filters in layers:
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.05)(x)
    return x

def classifyLayers(x):
    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dense(16)(x)
    x = ReLU()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    return x

####################################################
def generate_arrays_from_h5(h5file, group, group_label, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    lineCnt = 0
    print("top of generate")
    while True:
        specs = h5file[group][batchcount:batchcount+batchsize].data
        labels = h5file[group_label][batchcount:batchcount+batchsize].data
        batchcount += 1
        if batchcount % 10 == 0:
            print("batchcount=", batchcount, "batchsize=", batchsize)
        yield (specs.obj, labels.obj)  #  Note Bene - for auto encoder target is input!

def makeDir(thisdir):
    try:
        os.mkdir(thisdir)
    except:
        print('Already have ', thisdir)

def save_obj(obj, name):
    #    print("in save_obj_", name)
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    #    print("in load_obj ", os.getcwd())
    with open(name, 'rb') as f:
        return pickle.load(f)

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

####################################################  RUN RUN RUN

#######################  User set parameters VVVVVVVVVVVVVVV below

baseDir = "/home/val/PycharmProjects/github/signal-classifier/"

loadAE_ModelFilename = ""
h5filename = "h5fileHWcalls_1110records.h5"

batchsize = 100
total_epochs = 10
prior_epochs = 0

encoderLayers = [256, 128, 32, 8]

#######################  User set parameters ^^^^^^^^^^^^^^^^^^^  above

newModelID = "Classifier_75_20_5_{}_Em_h5".format(encoderLayers)
newModelID = newModelID.replace(", ", "-")

saveClassifier_ModelDir = "models/{}_{}_{}_epochs/".format(newModelID, prior_epochs, total_epochs)
makeDir(saveClassifier_ModelDir)


# Instantiate the Model
if loadAE_ModelFilename != "":
    classifyAE_Model = keras.models.load_model(loadAE_ModelFilename)
else:
    ##########################  the Model
    # Setup Encoder taking input spectrograms and outputting feature sets at the bottleneck layer
    # Setup classifyLayers taking bottleneck features and predicting label
    # The input tensor
    inputs = Input(shape=(256, 256, 1))
    # The encoder
    bottleneck = encoderAE(inputs, encoderLayers)
    encoder_Model = Model(inputs, bottleneck)
    # The classifier
    outputs = classifyLayers(bottleneck)
    classifyAE_Model = Model(inputs, outputs)

classifyAE_Model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4),
                             metrics=['accuracy'])  # adam default learning_rate is 0.001

print(classifyAE_Model.summary())
# this plots to the directory where the .py file is
plot_model(classifyAE_Model,show_shapes=True,show_dtype=True,expand_nested=True, to_file='models/{}.png'.format(newModelID))

### setup the data input
# initialize generator
h5file = h5py.File(h5filename, 'r')
print(h5filename, "keys are", h5file.keys())
for key in h5file.keys():
    print(key, "length", len(h5file[key]))
train = generate_arrays_from_h5(h5file, 'train_specs', 'train_labels', batchsize)  # train is (spectrograms, labels)
test = generate_arrays_from_h5(h5file, 'test_specs', 'test_labels', batchsize)
eval = generate_arrays_from_h5(h5file, 'eval_specs', 'eval_labels', batchsize)
printConfusionMatrix(classifyAE_Model, "train dataset", next(train))
#data = next(train)
#  data[0] are specs  data[1] are labels
chkptFile = "models/" + "{}_best_1100.ckpt".format(newModelID)

checkpoint = ModelCheckpoint(chkptFile, monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)

########## Run training epochs
tstart = time.time()
print('start model at time', tstart, "secs")
history = classifyAE_Model.fit(train, steps_per_epoch=10, initial_epoch=prior_epochs,epochs=total_epochs, verbose=2,\
                         validation_data=test, validation_steps=10, callbacks=[checkpoint])

save_obj(history, baseDir+"models/history_{}_[]-{}_1100.pkl".format(newModelID, prior_epochs, total_epochs))
#########################################  Done running model

print("save classifyAE_Model {} at directory \n{}".format(newModelID, saveClassifier_ModelDir))
classifyAE_Model.save(saveClassifier_ModelDir)


tstop = time.time()
print("Elapsed time s {}, m {}, hr {:.2f} s/epoch {:.2f} ".format(int(tstop - tstart), int((tstop - tstart) / 60.0),
                                                                  ((tstop - tstart) / 3600.0),
                                                                  (tstop - tstart) / (total_epochs - prior_epochs)))
##########################

# list all data in history
print("history keys",history.history.keys(), "models/history_{}_[]-{}_1100.pkl".format(newModelID, prior_epochs, total_epochs))
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

###  print confusion matrix on the data files
printConfusionMatrix(classifyAE_Model, "train dataset", next(train))
printConfusionMatrix(classifyAE_Model, "test dataset", next(test))
printConfusionMatrix(classifyAE_Model, "eval dataset", next(eval))

h5file.close()

############################################################
# This routine loads all the trained MLPs from the ensemble and lets them make weighted predictions
# What we need:
#   Load all MLPs
#   Plot error during training for each of them
#   Load weights and test data
#   Make weighted predictions
#   Filepath of all necessary files /Users/mh/Documents/CSML/IRDM/GroupCW/Bitbucket/archive2
############################################################

import numpy as np
import pickle
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam

############################################################
sys.path.append('/Users/mh/Documents/CSML/IRDM/GroupCW/Bitbucket/archive2/')

nFeat=41


def construct_mlp(saved_weights, modelStructure=None, dropOuts=None, nFeatures=nFeat):
    # construct MLP and set the layer weights to the saved weights

    if modelStructure is None:
        modelStructure = [500, 100, 20]

    if dropOuts is None:
        dropOuts = [0.8, 0.5, 0.5]

    model = Sequential()

    hiddenLayer1 = Dense(output_dim=500, activation='relu', input_dim=nFeat, trainable=False)
    hiddenLayer1.set_weights(saved_weights[0])
    #dropOut1 = Dropout(p=0.8)


    hiddenLayer2 = Dense(output_dim=100, activation='relu', input_dim=500, trainable=False)
    hiddenLayer2.set_weights(saved_weights[1])
    #dropOut2 = Dropout(p=0.5)


    hiddenLayer3 = Dense(output_dim=20, activation='relu', input_dim=100, trainable=False)
    hiddenLayer3.set_weights(saved_weights[2])
    #dropOut3 = Dropout(p=0.5)


    outputLayer = Dense(1, input_dim=20)
    outputLayer.set_weights(saved_weights[3])

    model.add(hiddenLayer1)
    #model.add(dropOut1)
    model.add(hiddenLayer2)
    #model.add(dropOut2)
    model.add(hiddenLayer3)
    #model.add(dropOut3)
    model.add(outputLayer)

    adam = Adam(lr=0.00001)
    model.compile(loss='mse', optimizer=adam)
    return model



if __name__ == "__main__":

    # LOAD AGAIN THE DIFFERENT DATASETS
    Y_Data = np.load("animesh/Y_Data.npy")
    Y_Data = Y_Data[~(Y_Data == 0).all(1)]
    X_mod_24 = np.load("X_Data_mod_24.npy") # ZERO COLUMNS, ZERO ROWS ALREADY CLEANED

    X_mod_48 = np.load("X_Data_mod_48.npy")
    X_mod_48 = np.delete(X_mod_48, [6,7,8,9,10,11,12,13,14,15,16,18,19,20,22,23,24,25,27,28,29],1)
    X_mod_48 = X_mod_48[~(X_mod_48==0).all(1)]

    X_mod_72 = np.load("X_Data_mod_72.npy")
    X_mod_72 = np.delete(X_mod_72, [6,7,8,9,10,11,12,13,14,15,16,18,19,20,22,23,24,25,27,28,29],1)
    X_mod_72 = X_mod_72[~(X_mod_72==0).all(1)]

    X_mod_96 = np.load("X_Data_mod_98.npy")
    X_mod_96 = np.delete(X_mod_96, [6,7,8,9,10,11,12,13,14,15,16,18,19,20,22,23,24,25,27,28,29],1)
    X_mod_96 = X_mod_96[~(X_mod_96==0).all(1)]

    # TEST DATA
    X_test_24 = X_mod_24[49952:, ]
    X_test_48 = X_mod_48[49952:, ]
    X_test_72 = X_mod_72[49952:, ]
    X_test_96 = X_mod_96[49952:, ]
    y_test = Y_Data[49952:]

    # create dictionary to test sets to make referencing easier
    test_dict = {0: (X_test_24, 0.025), 1: (X_test_48, 0.92499999), 2: (X_test_72, 0.025), 3: (X_test_96, 0.02500001)}
    # in pred we will accumulate the predictions from each MLP
    pred = np.zeros((y_test.shape[0],1))
    # path to fine tuned weights
    path = "/Users/mh/Documents/CSML/IRDM/GroupCW/Bitbucket/archive3/"

    mlp_list = []
    w_list = []
    for idx in range(4):
        fname = "fine_tune_weights_mlpEnsemble_" + idx.__str__()
        sWeights = pickle.load(open(path + fname, "rb"))
        w_list.append(sWeights)
        Mlp = construct_mlp(saved_weights=sWeights)
        mlp_list.append(Mlp)
        cpred = Mlp.predict(test_dict[idx][0], batch_size=10, verbose=1)
        pred += test_dict[idx][1] * cpred

    print pred










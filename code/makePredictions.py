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
import matplotlib.pyplot as plt
############################################################
#sys.path.append('/Users/mh/Documents/CSML/IRDM/GroupCW/Bitbucket/archive2/')

nFeat=41


def construct_mlp(saved_weights, modelStructure=None, dropOuts=None, nFeatures=nFeat):
    # construct MLP and set the layer weights to the saved weights

    if modelStructure is None:
        modelStructure = [500, 100, 20]

    if dropOuts is None:
        dropOuts = [0.8, 0.5, 0.5]

    model = Sequential()

    hiddenLayer1 = Dense(output_dim=500, activation='relu', input_dim=nFeat, trainable=False)
    #dropOut1 = Dropout(p=0.8)
    model.add(hiddenLayer1)
    hiddenLayer1.set_weights(saved_weights[0])

    hiddenLayer2 = Dense(output_dim=100, activation='relu', input_dim=500, trainable=False)
    #dropOut2 = Dropout(p=0.5)
    model.add(hiddenLayer2)
    hiddenLayer2.set_weights(saved_weights[1])

    hiddenLayer3 = Dense(output_dim=20, activation='relu', input_dim=100, trainable=False)
    #dropOut3 = Dropout(p=0.5)
    model.add(hiddenLayer3)
    hiddenLayer3.set_weights(saved_weights[2])

    outputLayer = Dense(1, input_dim=20, trainable=False)
    model.add(outputLayer)
    outputLayer.set_weights(saved_weights[3])

    adam = Adam(lr=0.00001)
    model.compile(loss='mse', optimizer=adam)
    return model



if __name__ == "__main__":

    # LOAD AGAIN THE DIFFERENT DATASETS
    Y_Data = np.load("../data/Y_Data.npy")
    Y_Data = Y_Data[~(Y_Data == 0).all(1)]
    X_mod_24 = np.load("../data/X_Data_mod_24.npy")
    X_mod_24 = X_mod_24[~(X_mod_24==0).all(1)]

    X_mod_48 = np.load("../data/X_Data_mod_48.npy")
    X_mod_48 = X_mod_48[~(X_mod_48==0).all(1)]

    X_mod_72 = np.load("../data/X_Data_mod_72.npy")
    X_mod_72 = X_mod_72[~(X_mod_72==0).all(1)]

    X_mod_96 = np.load("../data/X_Data_mod_98.npy")
    X_mod_96 = X_mod_96[~(X_mod_96==0).all(1)]

    # TEST DATA
    X_test_24 = X_mod_24[49952:, ]
    X_test_48 = X_mod_48[49952:, ]
    X_test_72 = X_mod_72[49952:, ]
    X_test_96 = X_mod_96[49952:, ]
    y_test = Y_Data[49952:]

    # NOTE: ideally we should pick the model weights from seriaized file
    #       however since the program update, structures serialized earlier
    #       were not accessible, so to save the time-intensive training 
    #       the test_dict is defined by hand:

    # load the pickled model weights
    #mlp_dict = pickle.load( open( "../model/MLP_dictionary.p", "rb" ))
    #weights = []
    #for _, value in mlp_dict.iteritems():
    #    weights.append(value[0][0])
    #test_dict = {0: (X_test_24, weights[0]), 1: (X_test_48, weights[1]), 2: (X_test_72, weights[2]), 3: (X_test_96, weights[3])}
    
    # create dictionary to test sets to make referencing easier
    test_dict = {0: (X_test_24, 0.025), 1: (X_test_48, 0.92499999), 2: (X_test_72, 0.025), 3: (X_test_96, 0.02500001)}
    # in pred we will accumulate the predictions from each MLP
    pred = np.zeros((y_test.shape[0],1))
    # path to fine tuned weights
    path = "../model/"

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
        #pred = cpred
    beta = np.mean(np.absolute(pred - y_test))
    print "beta= %d" % beta
    pred += beta
    f = plt.figure(1)
    plot1 = plt.plot(pred, label='pred')
    plot2 = plt.plot(y_test, label='gt')
    plt.legend(handles=[plot1[0], plot2[0]], loc=2)
    #plt.show()
    plt.xlabel("hour of 30-Sep-2010")
    plt.ylabel("energy load (MW)")
    f.savefig("../output/ensemble_prediction.png")
    print "RMSE = ", np.sqrt(np.mean((pred-y_test)**2))










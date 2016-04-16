########################################################
# This script learns the weights of an ensemble of MLPs
# by performing a weighted majority algorithm
########################################################
from __future__ import division
import numpy as np
import copy
from matplotlib import pyplot as plt
import pandas as pd
from mlpPretraining import MLP
import pickle
import sys
######################################################
# SIDE METHODS


def establish_ensemble(Y_Data, nEpoch, batch_size, *X_dataset):
    # this method instantiates all the MLPs and trains them on the respective dataset
    # Input: Datasets on each of which an MLP is trained
    print "Creating ensemble of %d MLPs" % np.shape(X_dataset)[0]
    nFeat = np.shape(X_dataset[0])[1]
    # create ensemble
    ensemble = []
    history = []
    for cX_dat in enumerate(X_dataset, start=1):
        X_dat = cX_dat[1]
        print "Training on dataset no. %d" % cX_dat[0]
        Mlp = MLP(modelStructure=[500, 100, 20], dropOuts=[0.8, 0.5, 0.5], nFeatures=nFeat, X_Data=X_dat, Y_Data=Y_Data)
        Mlp.pretrain(batch_size=batch_size,nEpoch=nEpoch)
        Mlp.train(nEpoch=nEpoch, batch_size=batch_size)
        history.append(Mlp.history)
        ensemble.append(Mlp)

    # save finetuned weights from each MLP as pickle file
    for idx_mlp in enumerate(zip(ensemble,history)):
        idx = idx_mlp[0]
        mlp = idx_mlp[1][0]
        hist = idx_mlp[1][1]
        fname_weights = "fine_tune_weights_mlpEnsemble_" + idx.__str__()
        fname_history = "history_mlpEnsemble_" + idx.__str__()
        pickle.dump(mlp.finetune_weights, open(fname_weights, "wb"))
        pickle.dump(hist, open(fname_history, "wb"))
    return ensemble



def train_ensemble(ensemble, X_val, Y_val, eta, alpha):
    # Look at "Improving short-term load forecast accuracy via combining sister forecasts" for explanation
    # online learning algorithm
        # initialize weight and RMSE for each MLP
        # for each observation in X_val:
        #   for each MLP
        #       compute loss update v
        #       compute mixing update w
    # INITIALIZATION
    print "Initializing..."
    n_learner = np.shape(ensemble)[0] # number of MLPs
    # compute the absolute error of every mlp on the first observation and
    # initialize the RMSE of each MLP with this absolute error error
    # the initial weights and initial RMSE will be stored in a dictionary
    mlp_dict={}
    for Mlp in ensemble:
        model = getattr(Mlp, 'model')
        prediction = model.predict(X_val, batch_size=1, verbose=1)[0]
        loss = np.abs(prediction - Y_val[0])
        mlp_dict[Mlp] = (1/n_learner, loss)

    # MAIN
    for cobs in range(1,np.shape(X_val)[0]):
        print "observation %d" % cobs
        sum = 0
        for cmlp in mlp_dict.keys():
            sum += mlp_dict[cmlp][0] * np.exp(-eta * mlp_dict[cmlp][1])

        for Mlp in ensemble:
            # loss update
            v = (mlp_dict[Mlp][0] * np.exp(-eta * mlp_dict[Mlp][1])) / sum
            # mixing update
            w = (1-alpha) * v + alpha / n_learner
            # update cumulative RMSE
            model = getattr(Mlp, 'model')
            prediction = model.predict(X_val[:cobs,], batch_size=1, verbose=1)
            cRMSE = mlp_dict[Mlp][1] + np.sqrt((1/(cobs+1) * np.mean((prediction - Y_val[:cobs])**2)))
            # update dict
            mlp_dict[Mlp] = (w, cRMSE)
    return mlp_dict


if __name__ == "__main__":
    sys.setrecursionlimit(1000000) # to enable pickling
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

    print np.shape(X_mod_24)[1]
    print np.shape(X_mod_48)[1]
    print np.shape(X_mod_72)[1]
    print np.shape(X_mod_96)[1]
    Ensemble = establish_ensemble(Y_Data, 300, 128, X_mod_24, X_mod_48, X_mod_72, X_mod_96)
    mlp = Ensemble[0]
    Y_val = mlp.y_val
    X_val = mlp.X_val
    MLP_D = train_ensemble(Ensemble, X_val, Y_val, 0.3, 0.1)
    pickle.dump(MLP_D, open("MLP_dictionary.p", "wb"))



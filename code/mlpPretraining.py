import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2
import copy
import pickle




class MLP(object):
    def __init__(self, modelStructure, dropOuts, nFeatures, X_Data, Y_Data):
        """
        :param modelStructure: e.g. [500, 100, 20] gives the number of units at each hidden layer
        :param dropOuts: e.g. [0.8, 0.8, 0.5] gives the dropout probabilities at each hidden layer
        """
        self.nFeatures = nFeatures
        self.modelStructure = modelStructure
        self.dropOuts = dropOuts
        self.X_Data = X_Data
        self.Y_Data = Y_Data
        idx_of_sep_30th_2010_hour0 = 49951 # first index of sep 30th 2010 in X_Data (at least so it is believed)

        self.X_train = X_Data[:49927, ]
        self.y_train = Y_Data[:49927]
        self.X_val = X_Data[49928:49951, ]
        self.y_val = Y_Data[49928:49951]
        self.X_test = X_Data[49952:, ]
        self.y_test = Y_Data[49952:]

        if np.shape(self.dropOuts) <> np.shape(self.modelStructure):
            raise IOError('number of dropout parameters must be equal number of hidden layers')

    def pretrain(self, batch_size=200, act='relu', nEpoch = 200):

        # we have to instantiate the model over and over again because we cannot freeze a layer after its been compiled
        self.weights = []
        for cSpecs in enumerate(zip(self.modelStructure, self.dropOuts), start=1):
            # cSpecs is e.g. (1, (500, 0.8)) where the first element gives how many hidden layers we have right now
            nLayers = cSpecs[0]
            nOut = cSpecs[1][0]
            pDrop = cSpecs[1][1]

            if nLayers == 1:
                # we have input -> hidden -> output
                model = Sequential()

                hiddenLayer = Dense(nOut, activation=act, input_dim=self.nFeatures) # first hidden layer
                dropoutLayer = Dropout(pDrop) # dropout layer
                outputLayer = Dense(1) # output layer (we have only one if nLayers == 0)

                model.add(hiddenLayer)
                model.add(dropoutLayer)
                model.add(outputLayer)

                # fit model
                adam = Adam(lr=0.00001)
                model.compile(loss='mse', optimizer=adam)
                model.fit(self.X_train, self.y_train, batch_size=batch_size, validation_split=0.2,
                            show_accuracy=True, verbose=1, nb_epoch=nEpoch)

                # save weights of hiddenLayer
                self.weights.append(hiddenLayer.get_weights())
                # instantiate a fresh model after grabbing the weights from the first hidden layer
                model = Sequential()

            else:
                # we have input -> hidden -> .. -> hidden -> output (at least two hidden layers)
                # nLayer indicates how many hidden layers we have (counting from 0)
                # first build frozen layers up to the current one
                for cLayer in range(nLayers):
                    nOut_cc = self.modelStructure[cLayer]
                    pDrop_cc = self.dropOuts[cLayer]

                    print "nlayers = ", nLayers
                    print "clayer = ", cLayer
                    print "layers.shape = ", len(self.weights)

                    if cLayer == 0:
                        # build first frozen layer
                        model.add(Dense(nOut_cc, weights=self.weights[cLayer], input_dim=self.nFeatures, activation=act, trainable=False))
                        model.add(Dropout(pDrop_cc))
                    elif cLayer < nLayers-1:
                        # build next frozen layer
                        model.add(Dense(nOut_cc, weights=self.weights[cLayer], activation=act, trainable=False))
                        model.add(Dropout(pDrop_cc))
                    else:
                        # build new hidden layer (which we want to train)
                        # Caveat: Now we have to use nOut # of hidden units and pDrop as dropout parameter
                        # as this is the hidden layer we add to the network
                        hiddenLayer = Dense(nOut, activation=act)
                        dropoutLayer = Dropout(pDrop)
                        outputLayer = Dense(1)

                        model.add(hiddenLayer)
                        model.add(dropoutLayer)
                        model.add(outputLayer)

                        # fit model
                        adam = Adam(lr=0.00001)
                        model.compile(loss='mse', optimizer=adam)
                        model.fit(self.X_train, self.y_train, batch_size=batch_size, validation_split=0.2,
                                    show_accuracy=True, verbose=1, nb_epoch=nEpoch)

                        # save weights of hiddenLayer
                        self.weights.append(hiddenLayer.get_weights())
                        model = Sequential()
        print "Pretraining complete"

    def train(self, act='relu', nEpoch=50, batch_size=200):
        # now train the entire mlp (without frozen layers)
        self.finetune_weights = []
        hiddenLayers = []
        self.model = Sequential()
        nLayer = np.shape(self.modelStructure)[0]
        for cSpecs in enumerate(zip(self.modelStructure, self.dropOuts)):
            cLayer = cSpecs[0]
            nOut = cSpecs[1][0]
            pDrop = cSpecs[1][1]
            if cLayer == 0:
                hiddenLayer = Dense(nOut, weights=self.weights[cLayer], input_dim=self.nFeatures, activation=act) #ACTIVATE THIS LINE WHEN INCLUDING PRETRAINING
                #hiddenLayer = Dense(nOut, input_dim=self.nFeatures, activation=act) # ACTIVATE THIS WHEN THERE IS NO PRETRAINING
                hiddenLayers.append(hiddenLayer)
                dropoutLayer = Dropout(pDrop)
                # self.finetune_weights.append(hiddenLayer.get_weights())
                self.model.add(hiddenLayer)
                self.model.add(dropoutLayer)

            elif cLayer < nLayer:
                hiddenLayer = Dense(nOut, weights=self.weights[cLayer], activation=act) #ACTIVATE THIS LINE WHEN INCLUDING PRETRAINIG
                #hiddenLayer = Dense(nOut, activation=act) # ACTIVATE THIS WHEN THERE IS NO PRETRAINING
                hiddenLayers.append(hiddenLayer)
                dropoutLayer = Dropout(pDrop)
                # self.finetune_weights.append(hiddenLayer.get_weights())
                self.model.add(hiddenLayer)
                self.model.add(dropoutLayer)
            """
            if cLayer == nLayer-1:
                outputLayer = Dense(1)
                self.model.add(outputLayer)
                hiddenLayers.append(outputLayer)
            """
        outputLayer = Dense(1)
        self.model.add(outputLayer)
        hiddenLayers.append(outputLayer)
        # fit model
        adam = Adam(lr=0.00001)
        self.model.compile(loss='mse', optimizer=adam)
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, validation_split=0.2,
                    show_accuracy=True, verbose=1, nb_epoch=nEpoch)
        for hL in hiddenLayers:
            self.finetune_weights.append(hL.get_weights())
        print "training complete"

def select_weather_station(X_Data, Y_Data, keepNoStations=1):
    # what happens to validation error if we leave out weather station k?
    # the weather station data is in columns 6:30
    stationsRemoved = []
    nFeat = np.shape(X_Data)[1]
    noStations = 25
    first_weather_index = 6
    last_weather_index = first_weather_index + noStations - 1
    wStationCols = [s for s in range(first_weather_index, last_weather_index)]
    print "Deleting worst weather station from %d remaining stations" % np.shape(wStationCols)[0]
    selecting = True
    while selecting:
        # go through every remaining weather station, fit model without one station and evaluate
        # delete station from wStationCols which gave highest evaluation error
        stationErrs = dict([(s, 0) for s in wStationCols])
        for s in wStationCols:
            # put zeros in the training matrix associated with weather station k
            Xcross = copy.deepcopy(X_Data)
            Xcross[:, s] = 0
            mlp = MLP(modelStructure=[500, 100, 20], dropOuts=[0.8, 0.5, 0.5], nFeatures=nFeat, X_Data=Xcross, Y_Data=Y_Data)
            mlp.pretrain()
            mlp.train()
            #stationErrs[s] = Mlp.model.evaluate(mlp.X_val, mlp.y_val)
            mdl = getattr(mlp, 'model')
            mlp_X_val = getattr(mlp, 'X_val')
            mlp_y_val = getattr(mlp, 'y_val')
            stationErrs[s] = mdl.evaluate(mlp_X_val, mlp_y_val)

            # free up Xcross
            Xcross = None
        # get station with largest validation error
        delStation = max(stationErrs, key=stationErrs.get)
        # set column of respective station to 0
        #X_Data[:, delStation] = 0
        stationsRemoved.append(delStation)
        # delete that station from the available stations list
        wStationCols.remove(delStation)
        # free up stationErrs
        stationErrs = None
        if np.shape(wStationCols)[0] <= keepNoStations: break
    return stationsRemoved



if __name__ == "__main__":

    X_Data = np.load("../data/X_Data.npy")
    Y_Data = np.load("../data/Y_Data.npy")
    X_Data = X_Data[~(X_Data == 0).all(1)]
    Y_Data = Y_Data[~(Y_Data == 0).all(1)]
    nFeat = np.shape(X_Data)[1]
    Mlp = MLP(modelStructure=[500, 100, 20], dropOuts=[0.8, 0.5, 0.5], nFeatures=nFeat, X_Data=X_Data, Y_Data=Y_Data)
    Mlp.pretrain(batch_size=200, nEpoch=300)
    Mlp.train(nEpoch=300)
    
    # from the training, learn which stations are irrelevant
    stationsRemoved = select_weather_station(X_Data, Y_Data, keepNoStations=3)
    
    # remove these from the saved feature matrices
    for slidebackto in [24, 48, 72, 98]:
        fname = "../data/X_Data_mod_" + str(slidebackto) + ".npy"
        X_Data = np.load(fname)
        X_Data = np.delete(X_Data, stationsRemoved, 1)
        np.save(fname, X_Data)
        print "adjusted stations in " + fname

    print "All done."

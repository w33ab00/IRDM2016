# IRDM2016 - Information Retrieval and Data Mining
## UCL group project on time-series forecasting

Team:  
+ Animesh Mishra
+ Konstantinos Metallinos
+ Maximilian Hoefl


Dataset: 
+ Global Energy Forecasting Competition 2014 ([details](http://www.drhongtao.com/gefcom))  
+ Download [link](http://1drv.ms/1PIVd0L).


Software dependencies:
+ Python 2.7.x and libraries (numpy, pandas, scikit-learn, matplotlib, keras, theano)  
  - Version compatibility: Keras 1.0.0 with Theano 0.8.0.dev0 as backend


Usage:

1. Get the dataset
  1. Extract the `GEFCom2014 Data/Load/Task 1/L1-train.csv` file from archive
  2. Put it in the `/data` folder
3. Go to `/code` folder. Run the python/R scripts in the following order:  
  <pre>
  dataVisualisation.py -> for visualizing the data
  Baseline_model.R     -> implements a baseline model (ARIMA) in R
  getFeatures.py       -> extracts features for python script
  mlpPretraining.py    -> pretraining the MLPs. Includes station selection.
  mlpEnsemble.py       -> trains the ensemble of MLPs
  makePredictions.py   -> predictions and plots
  </pre>


Folder description:  
+ `code`: has python/R scripts
+ `data`: has original data as well as data generated during feature processing
+ `model`: has parameters from model (e.g. during pretraining or final modeling)
+ `output`: has output figures
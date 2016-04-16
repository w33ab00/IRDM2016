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
  1.Extract the `GEFCom2014 Data/Load/Task 1/L1-train.csv` file from archive  
  2.Put it in the `/data` folder  
2. Go to `/code` folder. Run the python scripts in the following order:
  <pre>
  getFeatures.py
  mlpPretraining.py
  mlpEnsemble.py
  makePredictions.py
  dataVisualisation.py
  </pre>


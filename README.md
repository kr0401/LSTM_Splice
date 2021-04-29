# LSTM_Splice

These are codes for an accepted article ""Splice Junction Identification using Long Short-Term Memory Neural Networks" for publication in journal of Current Genomics. Please cite the article when using the works. In case you have ideas for future improvements and developments, contact the corresponding author. Thank you.

In order to run the codes, you will need Python 3.7 or above and TensorFlow 2.3.0 or above

To run the codes, 
1. Download the "full_data.csv" data, 
2. Download the "lstm-splice-10fCV.py" code
3. Place both the data and codes into your working directory, default is "C:/Users/abolf/.spyder-py3/full_data.csv", you will need to change abolf to the name of your OS username and change line 15 of the codes accordingly. 
4. Open the lstm-splice-10fCV.py in your python enterpretter and compile it, you may complie everything all at once or block by block.  

* This runs the model using 10-fold cross validation, each fold took on average 112 seconds to train the model and generate results in our Windows 10 OS with Corei7-10750H CPU and 32 GH RAM; total waiting time around 18 minutes. For systems with lower specifications, waiting time will be longer.  

** Outputs include values of Accuracy, F-Score, Tarining Time, Macro_auROC_OVO, Macro_auROC_OVR, Weighted_auROC_OVO, Weighted_auROC_OVR, and Macro_auPRC for each of the 10-folds. These values will be saved in a .csv file in your working directory, and a summary statistics of each goodness of fit measure will be generated upon successful run of the codes.  

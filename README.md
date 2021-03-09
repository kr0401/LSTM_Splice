# LSTM_Splice

This is early privilaged access to full codes and data for an artcile that is under review in the Journal of Current Genomics. Please hold back from share the contents untill the work has been accepted for publication. 

In order to run the codes, you will need Python 3.7 or above and TensorFlow 2.3.0 or above

To run the codes, 
1. Download the "full_data.csv" data, 
2. Download the "lstm-10fCV.py" codes
3. Place both the data and codes into your working directory, default is "C:/Users/abolf/.spyder-py3/full_data.csv", you will need to change abolf to the name of your OS username and change line 14 of the codes accordingly. 
4. Open the lstm-10fCV.py in your python enterpretter and compile it, you may complie everything all at once or block by block. 

* This runs the model using 10-fold cross validation, each fold takes on average 112 seconds to train the model and generate results on our Windows 10 OS with Corei7-10750H CPU and 32 GH RAM; total waiting time around 18 minutes. For systems with lower specifications, waiting time will be longer.

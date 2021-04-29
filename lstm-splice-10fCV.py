# Last Update on April 29, 2021

import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import time
from sklearn.model_selection import KFold

# Reading data, place data and this file on the following directory
data = pd.read_csv("C:/Users/USER/.spyder-py3/splicejunc/full_data.csv", dtype='unicode')
df = pd.DataFrame(data)

# changing labels unto codes
df = df.replace('EI', 0.0)
df = df.replace('IE', 1.0)
df = df.replace('N', 2.0)

# Length of sequence 60 is applied to longer sequences
for i in range(len(df['Sequence'])):
    if len(df['Sequence'][i]) == 60:
        df['Sequence'].iloc[i] = df['Sequence'].iloc[i][:59] #39
    elif len(df['Sequence'][i]) == 140:
        df['Sequence'].iloc[i] = df['Sequence'].iloc[i][40:99] #99

# Breaking sequences of length 60 to 20 3-letters sequences
l = df['Sequence'].tolist()
new_l = []
for seq in l:
    # Use for 3 base groupings 'CCA GCT...'
    new_l.append(' '.join([seq[i:i+3] for i in range(0, len(seq), 3)]))

# Adding the 3-letter words into data in a new column
df['Sequence2'] = new_l

# Shuffling DF randomly 
df = df.sample(frac=1).reset_index(drop=True)

# Seperating sequences from labels
label_list = df['Class'].tolist()
seq_list = df['Sequence2'].tolist()

label_arr = np.array(label_list)       # change to numpy array 
seq_arr = np.array(seq_list)           # change to numpy array 

# Tokenizing different 3-letter entries
tokenizer = Tokenizer(num_words=100)  # oov_token='<OOV>'
tokenizer.fit_on_texts(seq_arr)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(seq_arr)
padded = pad_sequences(sequences, maxlen=None)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

# Empty lists for saving results
accuracies = []
f_scores = []
times = []
macro_roc_auc_ovo = []
weighted_roc_auc_ovo = []
macro_roc_auc_ovr = []
weighted_roc_auc_ovr = []
macro_prc_auc = []

# Define the k-fold Cross Validator, k is set to 10, using a seed
kfold = KFold(n_splits=10, shuffle=True, random_state=1554356)

# Perfomring k-fold Cross Validation, model evaluation
fold_no = 1
for train, test in kfold.split(padded, label_arr):
    
    # Model Structure
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=18, input_length=20),
        tf.keras.layers.LSTM(26, return_sequences=True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(26, return_sequences=True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(26),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(26, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])

    #model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy']) 
    
    print('------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    start_time = time.time()
    history = model.fit(padded[train], label_arr[train], 
                    epochs=100, batch_size=50,
                    validation_data=(padded[test], label_arr[test]), 
                    verbose=0, callbacks=[callback])
    
    train_time = time.time()-start_time
    print('\nTraining Time:\t' + str(train_time) + '\n')
    times.append(train_time)      # storing computation times
    
    # Predictions in train and test sets of each fold
    #y_pred_train = model.predict(x_train)
    #cm_train = tf.math.confusion_matrix(labels=y_train, predictions=y_pred_train.argmax(axis=1)).numpy()
    
    y_pred_test = model.predict(padded[test])
    cm_test = tf.math.confusion_matrix(labels=label_arr[test], predictions=y_pred_test.argmax(axis=1)).numpy()
    
    recall = np.diag(cm_test) / np.sum(cm_test, axis = 1)
    precision = np.diag(cm_test) / np.sum(cm_test, axis = 0)
    ei_recall, ie_recall, n_recall, all_recall = recall[0], recall[1], recall[2], np.mean(recall)
    ei_precision, ie_precision, n_precision, all_precision = precision[0], precision[1], precision[2], np.mean(precision)

    ei_fscore = 2*((ei_precision*ei_recall)/(ei_precision+ei_recall))
    ie_fscore = 2*((ie_precision*ie_recall)/(ie_precision+ie_recall))
    n_fscore = 2*((n_precision*n_recall)/(n_precision+n_recall))
    all_fscore = 2*((all_precision*all_recall)/(all_precision+all_recall))

    # accuracy and f-score 
    accuracy = np.trace(cm_test) / np.sum(cm_test)
    accuracies.append(accuracy)
    f_scores.append(all_fscore)
    
    # auROC
    macro_roc_auc_ovo_t = roc_auc_score(label_arr[test], y_pred_test, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo_t = roc_auc_score(label_arr[test], y_pred_test, multi_class="ovo", average="weighted")
    macro_roc_auc_ovr_t = roc_auc_score(label_arr[test], y_pred_test, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr_t = roc_auc_score(label_arr[test], y_pred_test, multi_class="ovr", average="weighted")
    macro_roc_auc_ovo.append(macro_roc_auc_ovo_t)
    weighted_roc_auc_ovo.append(weighted_roc_auc_ovo_t)
    macro_roc_auc_ovr.append(macro_roc_auc_ovr_t)
    weighted_roc_auc_ovr.append(weighted_roc_auc_ovr_t)
    
    # auPRC
    precision = dict()
    recall = dict()
    average_precision = dict()
    y_bin = label_binarize(label_arr[test], classes=[0, 1, 2])

    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_pred_test[:, i])
        average_precision[i] = average_precision_score(y_bin[:, i], y_pred_test[:, i])
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), y_pred_test.ravel())
        average_precision["micro"] = average_precision_score(y_bin, y_pred_test, average="micro")
        
    macro_prc_auc.append(average_precision["micro"])
    
    keras.backend.clear_session()     # clearing memory of learned weights

    fold_no = fold_no + 1

# saving all fold accuracy, f-score, run time into a csv file
lstm_10fCV_gof = pd.DataFrame()
lstm_10fCV_gof['Accuracy'] = accuracies
lstm_10fCV_gof['F_Scores'] = f_scores
lstm_10fCV_gof['Train_Time'] = times
lstm_10fCV_gof['Macro_auROC_OVO'] = macro_roc_auc_ovo
#lstm_10fCV_gof['Macro_auROC_OVR'] = macro_roc_auc_ovr
#lstm_10fCV_gof['Weighted_auROC_OVO'] = weighted_roc_auc_ovo
#lstm_10fCV_gof['Weighted_auROC_OVR'] = weighted_roc_auc_ovr
lstm_10fCV_gof['Macro_auPRC'] = macro_prc_auc
lstm_10fCV_gof.to_csv('lstm_10fCV_gof.csv', index=False)

# printing summary values of accuracy, f-score, run time over 10-folds
lstm_10fCV_gof.describe()


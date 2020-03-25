#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:59:46 2020

@author: Christo Strydom
"""
import tensorflow as tf
from tensorflow import keras

import os
os.chdir('/media/lnr-ai/christo/github_repos/eximia/')
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
#%%
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%% This data comes from dukas_dataprep.py
eximia_candles_filename = 'data/USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_wrangled.csv'
eximia_df=pd.read_csv(eximia_candles_filename)

#%% This data comes from dl_dataprep.py
# fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_df'
fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_imaged'
image_dict=pd.read_pickle("data/{fname}.pkl".format(fname=fname))

output_df=pd.DataFrame()
for key in image_dict:
   if image_dict[key]:
      print(key)
      dict_df={}
      d=image_dict[key]
      dict_df['datetime']=d['datetime']
      dict_df['index']=d['index']
      # dict_df['datetime']=image_dict[key]['datetime']
      for count,ele in enumerate(d['close_image']): 
         dict_df['close_image_ema_{count}'.format(count=count+1)]=ele
      for count,ele in enumerate(d['low_image']): 
         dict_df['low_image_ema_{count}'.format(count=count+1)]=ele
      for count,ele in enumerate(d['high_image']): 
         dict_df['high_image_ema_{count}'.format(count=count+1)]=ele
      output_df = output_df.append(dict_df, ignore_index=True)
# image_dict[:10]   
# list(output_df)
output_df.head()
fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_output_df.csv'
output_df.to_csv(path_or_buf='data/{fname}'.format(fname=fname), index=False)

output_df=pd.read_csv(filepath_or_buffer='/media/lnr-ai/christo/github_repos/eximia/data/{fname}'.format(fname=fname))

output_df[list(output_df)].describe()
{k:image_dict[k] for k in range(1000)}

neg, pos = np.bincount(output_df['response'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# ['datetime',
#  'date',
#  'Gmt time',
#  'Open',
#  'High',
#  'Low',
#  'Close',
#  'Volume',
#  'bidopen',
#  'bidclose',
#  'bidhigh',
#  'bidlow',
#  'askopen',
#  'askclose',
#  'askhigh',
#  'asklow',
#  'tickqty',
#  'open',
#  'high',
#  'low',
#  'close',
#  'period',
#  'dom',
#  'awdn',
#  'month',
#  'doy',
#  'wny',
#  'timestamp',
#  'ny_timestamp',
#  'london_timestamp',
#  'jhb_timestamp',
#  'int_seconds',
#  'universaldate',
#  'is_business_day',
#  'previous_business_day',
#  'five_previous_business_day']

cleaned_df = eximia_df[['datetime', 
                        'close',
                        'int_seconds',
                        'universaldate',
                        'dom', 
                        'is_business_day']].merge(output_df,left_index=True, right_on='index')

# You don't want the `Time` column.
cleaned_df.pop('datetime_x')
cleaned_df.pop('datetime_y')
cleaned_df.pop('index')
cleaned_df.pop('trade_count')

cleaned_df_list=list(cleaned_df)
exc_list=['index','universaldate','dom','is_business_day','int_seconds',
 'response',
 'trade_count','close']
div_list=list(set(cleaned_df_list)-set(exc_list))
cleaned_df[div_list]= np.log(cleaned_df[div_list].div(cleaned_df.close, axis=0))
cleaned_df.pop('close')
cleaned_df['dom']=cleaned_df['dom']/31
cleaned_df['universaldate']=cleaned_df['universaldate']/35
cleaned_df['int_seconds']=cleaned_df['int_seconds']/86100
#%%
# np.log(cleaned_df[div_list])
# The `Amount` column covers a huge range. Convert to log-space.
# eps=0.001 # 0 => 0.1¢
# cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('response'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('response'))
test_labels = np.array(test_df.pop('response'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)
#%%

# from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

#%%

pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)

sns.jointplot(pos_df['ema_close_10'], pos_df['ema_close_20'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
plt.suptitle("Positive distribution")

sns.jointplot(neg_df['ema_close_10'], neg_df['ema_close_20'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
_ = plt.suptitle("Negative distribution")

#%%
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)),      
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model

#%%
  
EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

#%%

model = make_model()
model.summary()

#%%
model.predict(train_features[:10])
#%%
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

#%%
initial_bias = np.log([pos/neg])
initial_bias
   
   
   
#%%
model = make_model(output_bias = initial_bias)
model.predict(train_features[:10])

#%%
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))
#%%

initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)


#%%
"""
Confirm that the bias fix helps
Before moving on, confirm quick that the careful bias initialization actually helped.

Train the model for 20 epochs, with and without this careful initialization, and compare the losses:
"""   

model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)

#%%
model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)

#%%
def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()
  
#%%
  
plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)
#%%
"""
The above figure makes it clear: In terms of validation loss, on this problem, 
this careful initialization gives a clear advantage.
"""
#%% Train the model

model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels))

#%%
"""
Check training history
In this section, you will produce plots of your model's accuracy and loss on 
the training and validation set. These are useful to check for overfitting, 
which you can learn more about in this tutorial.

Additionally, you can produce these plots for any of the metrics you created 
above. False negatives are included as an example.

"""

def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
#%%
    
plot_metrics(baseline_history)

#%%
"""
Evaluate metrics
You can use a confusion matrix to summarize the actual vs. predicted labels 
where the X axis is the predicted label and the Y axis is the actual label.
"""

train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

#%%

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Trade  Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Trade decisions missed(False Negatives): ', cm[1][0])
  print('Trade signals correctly detected (True Positives): ', cm[1][1])
  print('Total trade signals: ', np.sum(cm[1]))
  
#%%
"""
Evaluate your model on the test dataset and display the results for the metrics you created above.
"""
baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline,0.17)
#%%
"""
Plot the ROC
Now plot the ROC. This plot is useful because it shows, at a glance, the range of performance the model can reach just by tuning the output threshold.

"""
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


#%%
"""
Class weights
Calculate class weights
The goal is to identify fradulent transactions, but you don't have very many of those positive samples to work with, so you would want to have the classifier heavily weight the few examples that are available. You can do this by passing Keras weights for each class through a parameter. These will cause the model to "pay more attention" to examples from an under-represented class.
"""

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

#%%
"""
Train a model with class weights
Now try re-training and evaluating the model with class weights to see how that affects the predictions.

Note: Using class_weights changes the range of the loss. This may affect the stability of the training depending on the optimizer. Optimizers whose step size is dependent on the magnitude of the gradient, like optimizers.SGD, may fail. The optimizer used here, optimizers.Adam, is unaffected by the scaling change. Also note that because of the weighting, the total losses are not comparable between the two models.
"""

weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels),
    # The class weights go here
    class_weight=class_weight) 

#%%

plot_metrics(weighted_history)

#%%
"""
Evaluate metrics
"""
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)

#%%
"""
Here you can see that with class weights the accuracy and precision are lower 
because there are more false positives, but conversely the recall and AUC are 
higher because the model also found more true positives. Despite having lower 
accuracy, this model has higher recall (and identifies more fraudulent 
transactions). Of course, there is a cost to both types of error (you wouldn't 
want to bug users by flagging too many legitimate transactions as fraudulent, 
either). Carefully consider the trade offs between these different types of 
errors for your application.
                                                                                                                                                                                                                                                                                                                                                                                   
"""                                                                                                                                                                                                                                                                                                                                                                                   
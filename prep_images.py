#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:40:52 2020

@author: Christo Strydom
"""
import numpy as np
import holidays
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
import holidays
#%%
# data, labels = np.arange(10).reshape((5, 2)), range(5)

# data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

us_holidays = holidays.UnitedStates()
'2020-01-01' in us_holidays 
za_holidays = holidays.SouthAfrica()
'2020-01-01' in za_holidays 
eximia_candles_filename = 'data/USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_wrangled.csv'
eximia_df=pd.read_csv(eximia_candles_filename)
list(eximia_df)

fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_imaged'
image_dict = pickle.load( open( "/home/lnr-ai/github_repos/eximia/data/{fname}.pkl".format(fname=fname), "rb" ) )  

# fullset=np.array()
fullset = np.empty([len(image_dict), 6, 10, 10])
response = np.empty([len(image_dict)])
valid_index_array=[]
for index in range(len(image_dict)):
   if image_dict[index]:
      stage = 'close'
      cimage = np.array(np.log(image_dict[index]['{stage}_image'.format(stage=stage)]/eximia_df.loc[index,stage]).reshape((10,10)))
      stage = 'high'
      himage = np.array(np.log(image_dict[index]['{stage}_image'.format(stage=stage)]/eximia_df.loc[index,stage]).reshape((10,10)))
      stage = 'high'
      limage = np.array(np.log(image_dict[index]['{stage}_image'.format(stage=stage)]/eximia_df.loc[index,stage]).reshape((10,10)))
      
      dom_image=np.ones(100).reshape((10,10))*eximia_df.loc[index,'dom']/31
      uni_image=np.ones(100).reshape((10,10))*eximia_df.loc[index,'universaldate']/35
      b_image=np.ones(100).reshape((10,10))*eximia_df.loc[index,'is_business_day']
      
      fimage=np.array([himage,limage,cimage,dom_image,uni_image,b_image])
      fullset[index,:]=fimage
      valid_index_array.append(index)
      response[index]=image_dict[index]['response']

data=fullset[np.array(valid_index_array),:] 
y=response[np.array(valid_index_array)]

# close_image=image_dict[index]['close_image'].reshape((10,10))
# high_image=image_dict[index]['high_image'].reshape((10,10))
# close_image=image_dict[index]['close_image'].reshape((10,10))
# eximia_df.loc[index].universaldate
# eximia_df.loc[index].close

# image.reshape((10,10))
      
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=42)

# import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(6, 10, 10)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

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
          input_shape=(6,10,10)),
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

model = make_model()
model.summary()
model.predict(x_train[:10])
# x_train[:10]

results = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))
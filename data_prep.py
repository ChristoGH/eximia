#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:17:33 2020

@author: Christo Strydom
"""


# import fxcmpy
import sys
sys.path.append("/media/lnr-ai/christo/github_repos/eximia/")
sys.path.append("/home/lnr-ai/github_repos/fxcm/")

import os
os.chdir('/media/lnr-ai/christo/github_repos/eximia/')
import pandas as pd
import logging
import numpy as np
import calendar
import datetime as datetime
from datetime import datetime as dt
from pandas import Timestamp
from pandas.tseries.offsets import BDay
from fxcm_timezone_lib import london_timestamp, ny_timestamp, jhb_timestamp
#from datetime import datetime
import sys
from data_prep_lib import eximia_wrangle_fn,MP_DUKA_DATETIME_FORMAT, duka_tick_prep_fn
from data_prep_lib import MP_FXCM_DATETIME_FORMAT
from data_prep_lib import pkl_dump, pkl_load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
#%%
data_path='/media/lnr-ai/christo/github_repos/eximia/data/'
bname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020'
#%%
image_name='{bname}_imaged'.format(bname=bname)
output_df_name='{bname}_output_df.csv'.format(bname=bname)
response_file_name='{bname}_response_df'.format(bname=bname)
wrangled_file_name='{bname}_wrangled'.format(bname=bname)
cleaned_file_name='{bname}_cleaned'.format(bname=bname)
train_file_name='{bname}_train'.format(bname=bname)
test_file_name='{bname}_test'.format(bname=bname)
val_file_name='{bname}_val'.format(bname=bname)

#%%The following should only be done ONCE.
eximia_candles_filename = '{data_path}{bname}'.format(bname=bname, data_path=data_path)
data=pd.read_csv(eximia_candles_filename+'.csv')
data=duka_tick_prep_fn(data=data, DATETIME_FORMAT= MP_DUKA_DATETIME_FORMAT)
eximia_df=eximia_wrangle_fn(data=data, DATETIME_FORMAT=MP_DUKA_DATETIME_FORMAT)
eximia_df.to_csv(eximia_candles_filename+'_wrangled'+ '.csv', index=False)

# Save to pickle
f = open("{data_path}{fname}.pkl".format(fname=wrangled_file_name,data_path=data_path),"wb")
pickle.dump(eximia_df,f)
f.close()  
eximia_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=wrangled_file_name, data_path=data_path), "rb" ) )  


#%%
eximia_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=wrangled_file_name, data_path=data_path), "rb" ) )  

eximia_df['gmt_datetime'] =  pd.to_datetime(eximia_df['Gmt time'], format=MP_DUKA_DATETIME_FORMAT)
eximia_df['datetime'] =  pd.to_datetime(eximia_df.date, format=MP_FXCM_DATETIME_FORMAT)
# eximia_df['date'] = pd.to_datetime(eximia_df.date, format=MP_FXCM_DATETIME_FORMAT).apply(f)

#%% prep np image arrays:
"""
Some notes:
and ema of history == 1 is trivial. It returns only the number.  force ema to 
start at history==2.  To generate a 10x10 image we need a 101 data points, 
starting at history==2 and ending at 101 (range(2,102) for 
"""
history=100
count=0
trade_count=0
eximia_df['response']=0
eximia_df['image']=0
image_dict={}
eximia_df['image'] = eximia_df['image'].astype(object)
for index, (entry_time, entry_high, entry_low, entry_close, exit_time, exit_high, exit_low, exit_close) in enumerate(zip(
                                                                     eximia_df['gmt_datetime'][0:-1],
                                                                    eximia_df.high[0:-1],
                                                                    eximia_df.low[0:-1],
                                                                    eximia_df.close[0:-1],
                                                                     eximia_df['gmt_datetime'][1:],                                                                    
                                                                    eximia_df.high[1:],
                                                                    eximia_df.low[1:], 
                                                                    eximia_df.close[1:])):
   count+=1
   capsule_dict={}
   if index>(history+1):
      close_series = eximia_df.loc[range(index-history,index+1)].close.values
      high_series = eximia_df.loc[range(index-history,index+1)].high.values
      low_series = eximia_df.loc[range(index-history,index+1)].low.values
      # series = eximia_df.loc[index-history:index].close.values # This returns a series of lenth history + 1
      # series=eximia_df[(index-history+1):(index+1)].close.values
      # print(len(series))
      close_image=ema_image_fn(close_series)
      high_image=ema_image_fn(high_series)
      low_image=ema_image_fn(low_series)
      capsule_dict['datetime']=entry_time
      capsule_dict['index']=index    
      capsule_dict['close_image']=close_image
      capsule_dict['high_image']=high_image   
      capsule_dict['low_image']=low_image   
      capsule_dict['response']=0   
   if (exit_close>entry_high) & (exit_low>entry_low) & ((exit_high-exit_low)>(entry_high-entry_low)):
      trade_count+=1
      print('trade at ', entry_time,' count=',count, 'trade count = ', trade_count, ' exit at ', exit_time)
      eximia_df.loc[index,'response']=1
      capsule_dict['response']=1  
   image_dict[index]=capsule_dict

# image_name='{bname}_imaged'.format(bname=bname)
f = open("{data_path}{fname}.pkl".format(fname=image_name,data_path=data_path),"wb")
pickle.dump(image_dict,f)
f.close()  
image_dict = pickle.load( open( "{data_path}{fname}.pkl".format(fname=image_name, data_path=data_path), "rb" ) )  

#%% The following creates the dataframe from the images and saves it to csv.
# fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_df'
# fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_imaged'
image_dict=pd.read_pickle("{data_path}{fname}.pkl".format(data_path=data_path,fname=image_name))

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
# fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_output_df.csv'
output_df.to_csv(path_or_buf='{data_path}{fname}'.format(fname=output_df_name, data_path=data_path), index=False)


f = open("{data_path}{fname}.pkl".format(fname=output_df_name,data_path=data_path),"wb")
pickle.dump(output_df,f)
f.close()  
output_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=output_df_name, data_path=data_path), "rb" ) )  
list(output_df)

#%% Here we create the responses, put the results into response_df and save to file
# eximia_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=wrangled_file_name, data_path=data_path), "rb" ) )  
eximia_df=pkl_load(fname=wrangled_file_name, data_path=data_path)
history=100
count=0
trade_count=0
eximia_df['response']=0
eximia_df['image']=0
image_dict={}
eximia_df['image'] = eximia_df['image'].astype(object)
response_df=pd.DataFrame()
datetimelist=[]
indexlist=[]
responselist=[]
for index, (entry_time, entry_high, entry_low, entry_close, exit_time, exit_high, exit_low, exit_close) in enumerate(zip(
                                                                     eximia_df['gmt_datetime'][0:-1],
                                                                    eximia_df.high[0:-1],
                                                                    eximia_df.low[0:-1],
                                                                    eximia_df.close[0:-1],
                                                                     eximia_df['gmt_datetime'][1:],                                                                    
                                                                    eximia_df.high[1:],
                                                                    eximia_df.low[1:], 
                                                                    eximia_df.close[1:])):
   count+=1
   capsule_dict={}
   response=0
   datetimelist.append(entry_time)
   indexlist.append(index)   
   if (exit_close>entry_high) & (exit_low>entry_low) & ((exit_high-exit_low)>(entry_high-entry_low)):
      trade_count+=1
      print('trade at ', entry_time,' count=',count, 'trade count = ', trade_count, ' exit at ', exit_time)
      # eximia_df.loc[index,'response']=1
      response=1  
   responselist.append(response)
   # response_df = response_df.append(capsule_dict, ignore_index=True)
   len(responselist)
response_df = pd.DataFrame(data={'datetime':datetimelist,'index':indexlist,'response':responselist})
datetimelist.append(entry_time)

pkl_dump(df=response_df, fname=response_file_name, data_path=data_path)
response_df = pkl_load(fname=response_file_name, data_path=data_path)
# response_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=response_file_name, data_path=data_path), "rb" ) )  
#%%
eximia_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=wrangled_file_name, data_path=data_path), "rb" ) )  
eximia_df['gmt_datetime'] =  pd.to_datetime(eximia_df['Gmt time'], format=MP_DUKA_DATETIME_FORMAT)
eximia_df['datetime'] =  pd.to_datetime(eximia_df.gmt_datetime, format=MP_FXCM_DATETIME_FORMAT)
eximia_df.pop('gmt_datetime')

output_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=output_df_name, data_path=data_path), "rb" ) )  
list(output_df)
output_df[list(output_df)].describe()
output_df['datetime'] =  pd.to_datetime(output_df.datetime, format=MP_FXCM_DATETIME_FORMAT)

response_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=response_file_name, data_path=data_path), "rb" ) )  
list(response_df)
response_df['datetime'] =  pd.to_datetime(response_df.datetime, format=MP_FXCM_DATETIME_FORMAT)

output_df[list(output_df)].describe()
# eximia_df['gmt_datetime'] =  pd.to_datetime(eximia_df['Gmt time'], format=MP_DUKA_DATETIME_FORMAT)

neg, pos = np.bincount(response_df['response'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

cleaned_df = eximia_df[['datetime', 
                        'close',
                        'int_seconds',
                        'universaldate',
                        'dom', 
                        'is_business_day']].merge(output_df,left_on='datetime', right_on='datetime')
cleaned_df = cleaned_df.merge(response_df,left_on=['datetime','index'], right_on=['datetime','index'])
# cleaned_df[['index_x','index_y']]
# response_df.head()
# output_df.head()
list(cleaned_df)
cleaned_df[['datetime','index']]

# You don't want the `Time` column.
cleaned_df.pop('datetime')
# cleaned_df.pop('gmt_datetime')
# cleaned_df.pop('datetime_y')
cleaned_df.pop('index')
# cleaned_df.pop('trade_count')

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
pkl_dump(df=cleaned_df, fname=cleaned_file_name, data_path=data_path)
cleaned_df = pkl_load(fname=cleaned_file_name, data_path=data_path)


train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

pkl_dump(df=train_df, fname=train_file_name, data_path=data_path)
pkl_dump(df=test_df, fname=test_file_name, data_path=data_path)
pkl_dump(df=val_df, fname=val_file_name, data_path=data_path)

train_df = pkl_load(fname=train_file_name, data_path=data_path)
train_df = pkl_load(fname=test_file_name, data_path=data_path)
val_df = pkl_load(fname=val_file_name, data_path=data_path)


# response_df = pickle.load( open( "{data_path}{fname}.pkl".format(fname=response_file_name, data_path=data_path), "rb" ) )  



# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('response'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('response'))
test_labels = np.array(test_df.pop('response'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

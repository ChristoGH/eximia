#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:17:51 2020

@author: Christo Strydom
"""
# import fxcmpy
import sys
sys.path.append("/home/lnr-ai/github_repos/fxcm/")
import os
os.chdir('/media/lnr-ai/christo/github_repos/eximia/')
import pandas as pd
# import logging
import numpy as np
# import calendar
import datetime as datetime
# from datetime import datetime as dt
# from pandas import Timestamp
# from pandas.tseries.offsets import BDay
# from fxcm_timezone_lib import london_timestamp, ny_timestamp, jhb_timestamp
#from datetime import datetime
import sys
# from datetime import datetime
MP_DUKA_DATETIME_FORMAT='%d.%m.%Y %H:%M:%S.%f'
MP_EXIMIA_DATETIME_FORMAT='%d.%m.%Y %H:%M:%S.%f'
MP_FXCM_DATETIME_FORMAT='%Y-%m-%d %H:%M:%S'
#%%
eximia_candles_filename = 'data/USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_wrangled.csv'
eximia_df=pd.read_csv(eximia_candles_filename)
# list(eximia_df)
# f=date_f()
# g=datestamp_f()
# eximia_df['gmt_datetime']=eximia_df['Gmt time'].apply(f)
eximia_df['gmt_datetime'] =  pd.to_datetime(eximia_df['Gmt time'], format=MP_DUKA_DATETIME_FORMAT)
# g=datetime.strptime(t, MP_EXIMIA_DATETIME_FORMAT)
eximia_df['datetime'] =  pd.to_datetime(eximia_df.date, format=MP_FXCM_DATETIME_FORMAT)
eximia_df['date'] = eximia_df['datetime'].apply(f) # day-of-the-year
eximia_df['date'] = pd.to_datetime(eximia_df.date, format=MP_FXCM_DATETIME_FORMAT).apply(f)
#%% Load user-defined functions

def datestamp_f(): return lambda x: datetime.strptime(x, MP_EXIMIA_DATETIME_FORMAT)
def date_f(): return lambda x: x.date()

def fxcm_date_fn(df):
   print('create date column...')
   f=date_f()
   df['date'] = df['datetime'].apply(f) # day-of-the-year
   return df

def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    # s = array(s)
    ema_list = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema_list.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema_list.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema_list[j]) * multiplier) + ema_list[j]
        j = j + 1
        ema_list.append(tmp)

    return ema_list


def ema_image_fn(series):
   image=np.zeros((100))
   # image.reshape((10,10))
   for n in range(100):
      # print(n)
      ema_output=ema(s=series, n=n+1)
      image[n]=ema_output[-1]
   return image

def ema_df_fn(series, name):
   output_dict={}
   # image.reshape((10,10))
   for n in range(100):
      # print(n)
      ema_output=ema(s=series, n=n+1)      
      output_dict[name+'_'+str(n+1)]=ema_output[-1]
   return(output_dict)

#%%-Verify the mechanics:

m=4
small_df=eximia_df[1000:1020]
for index, (entry_time, entry_close) in enumerate(zip(small_df['gmt_datetime'][0:-1],
                                                                    small_df.close[0:-1])):
   print('----------------------------------------')   
   print(index, entry_time, entry_close)
   if index>=(m-1):
      series=small_df[(index-m+1):(index+1)].close.values

      print(index, entry_time, entry_close)      
      print(series)
      
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
#%% Save images to a pickle file:
import pickle
fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_imaged'
f = open("/media/lnr-ai/christo/github_repos/eximia/data/{fname}.pkl".format(fname=fname),"wb")
pickle.dump(image_dict,f)
f.close()  
image_dict = pickle.load( open( "data/{fname}.pkl".format(fname=fname), "rb" ) )  
#%% prep market dataframes:
"""
Some notes:
We need 100 data points to create an 10x10 image
and ema of history == 1 is trivial. It returns only the number.  force ema to 
start at history==2.  To generate a 10x10 image we need a 101 data points, 
starting at history==2 and ending at 101 (range(2,102) for 
"""
history=100
count=0
trade_count=0
# eximia_df['response']=0
# eximia_df['image']=0
image_dict={}
# eximia_df['image'] = eximia_df['image'].astype(object)
output_df=pd.DataFrame()
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
      output_dict=ema_df_fn(series=high_series, name='ema_high')
      output_dict.update(ema_df_fn(series=low_series, name='ema_low'))
      output_dict.update(ema_df_fn(series=close_series, name='ema_close'))
      output_dict.update([ ('datetime', entry_time), 
                           ('index', index),
                           ('response', 0),
                           ('trade_count',trade_count)])   
      if (exit_close>entry_high) & (exit_low>entry_low) & ((exit_high-exit_low)>(entry_high-entry_low)):
         trade_count+=1
         print('trade at ', entry_time,' count=',count, 'trade count = ', trade_count, ' exit at ', exit_time)
         eximia_df.loc[index,'response']=1
         output_dict.update([ ('datetime', entry_time), 
                              ('index', index),
                              ('response', 1),
                           ('trade_count',trade_count)])
      output_df = output_df.append(output_dict, ignore_index=True)
   # image_dict[index]=capsule_dict
fname='USDZAR_Candlestick_5_M_BID_01.01.2019-08.02.2020_df'
output_df.to_pickle("/home/lnr-ai/github_repos/eximia/data/{fname}.pkl".format(fname=fname))
# import pyarrow
output_df.to_feather("/home/lnr-ai/github_repos/eximia/data/{fname}.feather".format(fname=fname))
#%%

image=np.zeros((100))
# image.reshape((10,10))
for n in range(100):
   print(n)
   ema_output=ema(s=series, n=n+1)
   image[n]=ema_output[-1]
# return image


#%% do a spot check:
# [	datetime	date	Gmt time	Open	High	Low	Close	Volume	bidopen	bidclose	bidhigh	bidlow	askopen	askclose	askhigh	asklow	tickqty	open	high	low	close	period	dom	awdn	month	doy	wny	timestamp	ny_timestamp	london_timestamp	jhb_timestamp	int_seconds	universaldate	is_business_day	previous_business_day	five_previous_business_day	gmt_datetime	response	image
# 79878	2019-10-09 00:00:00	2019-10-09	09.10.2019 07:25:00.000	15.20943	15.212229999999998	15.201529999999998	15.20459	458.74	15.20943	15.20459	15.212229999999998	15.201529999999998	15.20943	15.20459	15.212229999999998	15.201529999999998	458.74	15.20943	15.212229999999998	15.201529999999998	15.20459	201910	9	Wed	10	282	40	2019-10-09 07:25:00	2019-10-09 03:25:00	2019-10-09 08:25:00	2019-10-09 09:25:00	26700	11.0	True	2019-10-08	2019-10-02	2019-10-09 07:25:00	1	0
# ]
eximia_df.loc[79878]
# Out[559]: 
# datetime                          2019-10-09 00:00:00
# date                                       2019-10-09
# Gmt time                      09.10.2019 07:25:00.000
# Open                                          15.2094
# High                                          15.2122
# Low                                           15.2015
# Close                                         15.2046
# Volume                                         458.74
# bidopen                                       15.2094
# bidclose                                      15.2046
# bidhigh                                       15.2122
# bidlow                                        15.2015
# askopen                                       15.2094
# askclose                                      15.2046
# askhigh                                       15.2122
# asklow                                        15.2015
# tickqty                                        458.74
# open                                          15.2094
# high                                          15.2122
# low                                           15.2015
# close                                         15.2046
# period                                         201910
# dom                                                 9
# awdn                                              Wed
# month                                              10
# doy                                               282
# wny                                                40
# timestamp                         2019-10-09 07:25:00
# ny_timestamp                      2019-10-09 03:25:00
# london_timestamp                  2019-10-09 08:25:00
# jhb_timestamp                     2019-10-09 09:25:00
# int_seconds                                     26700
# universaldate                                      11
# is_business_day                                  True
# previous_business_day                      2019-10-08
# five_previous_business_day                 2019-10-02
# gmt_datetime                      2019-10-09 07:25:00
# response                                            1
# image                                               0
# Name: 79878, dtype: object

image=ema_image_fn(series)
seriesA=c
len(seriesA)
ema_output=ema(s=seriesA, n=1)

# Which is the same as:
seriesB=eximia_df.loc[79878-history:79878].close.values
len(seriesB)

seriesC=eximia_df.loc[range(79878-history,79878+1)].close.values
len(seriesC)
ema_image_fn(seriesC) 
p=100
ema(s=seriesC, n=p)[-1]
image_dict[79878]['close_image'][p-2]



seriesA==seriesB
[seriesA[i]==seriesB[i] for i in range(len(seriesA))]
eximia_df.loc[79878].close #=15.20459
series[-1] # = 15.20459

"""on 09.10.2019 07:25:00.000 GMT, USDZAR closed at 15.2046. Counting 100 5min 
candles back brings us to 08.10.2019 23:10:00.000 GMT
"""
image_dict[79878]['close_image'][9]
# series=eximia_df[(index-history):(index+1)].close.values
eximia_df.loc[(79878-history+1)]
# datetime                          2019-10-08 00:00:00
# date                                       2019-10-08
# Gmt time                      08.10.2019 23:10:00.000
# Open                                          15.2693
# High                                          15.2883
# Low                                           15.2684
# Close                                         15.2753
# Volume                                         121.75
# bidopen                                       15.2693
# bidclose                                      15.2753
# bidhigh                                       15.2883
# bidlow                                        15.2684
# askopen                                       15.2693
# askclose                                      15.2753
# askhigh                                       15.2883
# asklow                                        15.2684
# tickqty                                        121.75
# open                                          15.2693
# high                                          15.2883
# low                                           15.2684
# close                                         15.2753
# period                                         201910
# dom                                                 8
# awdn                                              Tue
# month                                              10
# doy                                               281
# wny                                                40
# timestamp                         2019-10-08 23:10:00
# ny_timestamp                      2019-10-08 19:10:00
# london_timestamp                  2019-10-09 00:10:00
# jhb_timestamp                     2019-10-09 01:10:00
# int_seconds                                     83400
# universaldate                                      10
# is_business_day                                  True
# previous_business_day                      2019-10-07
# five_previous_business_day                 2019-10-01
# gmt_datetime                      2019-10-08 23:10:00
# response                                            0
# image                                               0
"""
Which is the SAME close as the FIRST element in the 100 price series or 
series[0]
"""
series[0]

ema(s=series, n=1)

image_dict[79878]['close_image'][10]
p=99
ema(s=series, n=p)[-1]
image_dict[79878]['close_image'][p-2]
#%%
image=np.zeros((100))
# image.reshape((10,10))
for n in range(3,103):
   ema_output=ema(s=series, n=n)
   image[n-3]=ema_output[-1]/series[-1]
image.reshape((10,10))

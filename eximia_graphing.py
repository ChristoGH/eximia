#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:15:26 2019

@author: Christo Strydom
"""
# import matplotlib
import matplotlib.pyplot as plt
import pickle
# import fxcmpy
import os
import pandas as pd
import logging
import numpy as np
# import calendar
import datetime as datetime
from datetime import timedelta, datetime as dt

# from pandas import Timestamp
# from pandas.tseries.offsets import BDay
from string import ascii_lowercase,ascii_uppercase
#from datetime import datetime
import sys
os.chdir('/home/lnr-ai/github_repos/fxcm/')
fxcm_candles_filename = 'data/fxcm_candles.csv'
log_filename = 'data/logging_fxcm_candles.log'
tick_symbol='USD/ZAR'
config_file='/home/lnr-ai/github_repos/fxcm/config/fxcm.cfg'
period='m30'
nhistory=250
MP_FXCM_DATETIME_FORMAT='%Y-%m-%d %H:%M:%S'
logging.basicConfig(format='%(asctime)s %(message)s', filename=log_filename, level=logging.DEBUG)
#%%
pre_market_list=['#']*14
post_market_list=['%']*4

def alpha_list_fn():
   # This function creates the alpha list, a list that assigns a
   # character to each 30 min period in a 24 hour day:
   alpha_list=[]   
   return_list=[] 
   for c in ascii_lowercase:
       alpha_list.append(c)
   for c in ascii_uppercase:
       alpha_list.append(c)    
   # alpha_list=alpha_list[:-3]
   # return_list.extend(pre_market_list)
   # return_list.extend(alpha_list[:-18])
   # return_list.extend(post_market_list)
   return_list=alpha_list
   print("Successfully created the alpha character list, length = {length}".format(length=len(return_list)))
   return return_list


def get_candle_history(tick_symbol, filename):
   candles_df=pd.DataFrame()
   try:
      candles_df = pd.read_csv(filepath_or_buffer=filename)
      candles_df.drop_duplicates(inplace=True)
      # candles_df.set_index('date')
      # candles_df.sort_values(by='date')
      candles_df.set_index('datetime', inplace=True)
      # store.close()
   except:
      logging.error('This error occurred in get_candle_history:{e}'.format(e=sys.exc_info()[0]))      
   return candles_df

def delta_fn(candles_df, increments, significant_figs):
   # typical: delta_fn(candles_df=candles_df, increments=100, significant_figs=5)
   max_df=candles_df[['date', 'high']].groupby(['date']).max()
   min_df=candles_df[['date', 'low']].groupby(['date']).min()
   span_df=max_df.merge(right=min_df, left_index=True, right_index=True)
   # mean(span_df.high-span_df.low)
   mean_range= np.mean((span_df.high-span_df.low).values)
   return np.round(mean_range/increments*10**significant_figs)/10**significant_figs

def alpha_fn(df, mp_period):
   index=df.mp_period==mp_period
   return df[index].alpha.values[0]

def update_mp_dict(mp_dict, mp_low, mp_high, alpha):
   mp_range=[int(m*delta*10000)/10000 for m in list(range(mp_low, mp_high))]
   for x in mp_range:
      if (x in list(mp_dict)):
         mp_dict[x]=mp_dict[x]+alpha
      else:
         mp_dict[x]=alpha
   return mp_dict

def mp_price_fn(price,open_point,delta):
   # mp_high = int(np.ceil((price-open_point)/delta))
   return int(np.ceil((price-open_point)/delta))
   
def mp_high_low_fn(open_point, low, high, delta):
   mp_high = mp_price_fn(high,open_point,delta)
   mp_low = mp_price_fn(low,open_point,delta)   
   # mp_high = int(np.ceil((high-open_point)/delta))
   # mp_low = int(np.ceil((low-open_point)/delta))
   return mp_low, mp_high

def high_low_fn(df,index):
   low = df[index].low.values[0]
   high = df[index].high.values[0]
   return low, high

def index_fn(df, mp_period):
   return df.mp_period==mp_period

def open_price_fn(df, mp_period, open_date):
   return df[df.date==open_date].open.values[0]
   
def sum_print_fn(mp_dict):
   # sum_print_list, poc_len = sum_print_fn(mp_dict)
   poc_len=0
   sum_prints=0
   sum_print_list=[]
   mp_prints={}
   # low_range_price=0
   for key in mp_dict:
      # print(key, mp_dict[key], len(mp_dict[key]),poc_len)
      poc_len=max(poc_len, len(mp_dict[key]))
      # print(sum_prints, len(mp_dict[key]), low_range, top_range)  
      sum_prints=sum_prints+len(mp_dict[key])
      mp_prints[key]=sum_prints
      sum_print_list.append(sum_prints)
   return sum_print_list, poc_len

def define_mp_range(totalprints, lower_percentage, higher_percentage):
   # low_range, top_range = define_mp_range(totalprints, lower_percentage, higher_percentage)
   low_range=int(totalprints*lower_percentage)
   top_range=int(totalprints*higher_percentage)
   return low_range, top_range

def range_index_fn(sumlist, range_value):
   return next(x for x, val in enumerate(sumlist) if val > range_value)
   
def range_price_fn(delta_list, index_value, delta):
   # delta_list[low_range_index]*delta
   return delta_list[index_value]*delta

def define_mp_pricerange(delta_list, sum_print_list, low_range, top_range, significant_figs):
   # low_range_price, top_range_price = define_mp_pricerange(delta_list, sum_print_list, low_range, top_range)
   d=10**significant_figs
   low_range_index = range_index_fn(sum_print_list, low_range)
   top_range_index = range_index_fn(sum_print_list, top_range)
   low_range_price = range_price_fn(delta_list, low_range_index, delta)
   top_range_price = range_price_fn(delta_list, top_range_index, delta)
   return int(low_range_price*d)/d, int(top_range_price*d)/d

def poc_fn(mp_dict,  poc_len):
   poc_list=[]
   for key in mp_dict:
      # print(key, mp_dict[key], len(mp_dict[key]),poc)
      if poc_len==len(mp_dict[key]):
         poc_list.append(key)
   return poc_list

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
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema
# delta_fn(candles_df=candles_df, increments=100, significant_figs=5)
   
#%%
# Collect data and do initial prep:
# Collect data:
candles_df= get_candle_history(tick_symbol=tick_symbol, filename=fxcm_candles_filename)
# Select only business days:
candles_df=candles_df[candles_df.is_business_day].copy()
# list(candles_df)
alpha_list=alpha_list_fn()
# candles_df['mp_period'].apply
candles_df.reset_index(inplace=True)
# Create the alpha column:
candles_df['alpha']=candles_df['mp_period'].apply(lambda x : alpha_list[x])
# Convert datatime column to datetime:
candles_df['datetime'] =  pd.to_datetime(candles_df.datetime, format=MP_FXCM_DATETIME_FORMAT)
# Convert the date column to date:
candles_df['date'] = candles_df['datetime'].apply(lambda x: x.date()) 
# Convert the previous business day column to datetime:
candles_df['previous_business_day'] =  pd.to_datetime(candles_df['previous_business_day'], format=MP_FXCM_DATETIME_FORMAT)
# define a list of unique dates in candles_df:
candles_df_date_list=list(set(list(candles_df['date'])))
# prevdate=list(candles_df['previous_business_day'])[-1]
# p = (candles_df['date']==prevdate).values
# q= (candles_df['mp_period']==0).values
# candles_df[p&q]['open'].values[0]

#%%

series=candles_df[0:1000].close.values
image=np.zeros((100))
# image.reshape((10,10))
for n in range(3,103):
   ema_output=ema(s=series, n=n)
   image[n-3]=ema_output[-1]/series[-1]
image.reshape((10,10))
# ```
# This piece of code is defective.  Don't run!
# ```
# candles_df_date_list=list(set(candles_df['date']))
# previous_business_date_dict={}
# remove_day_list=[]
# for d in candles_df_date_list:
#    print(d)
#    day_df=candles_df[candles_df.date==d]
#    previous_business_day_list=list(day_df['previous_business_day'])
#    previous_business_day=list(set(previous_business_day_list))[0]
#    previous_business_date=previous_business_day.date()
#    print(previous_business_date)
#    if (not candles_df[candles_df.date==previous_business_date].empty):
#       previous_business_date_dict[d]=previous_business_date
#    else:
#       remove_day_list.append(d)
      
# candles_df=candles_df[candles_df.date!=remove_day_list[0]].copy()
# print(candles_df.shape)
   # pd.to_datetime(list(set(list(candles_df[candles_df.date==d]['previous_business_day'])))[0], format=MP_FXCM_DATETIME_FORMAT).date()
   # print(pd.to_datetime(list(set(list(candles_df[candles_df.date==d]['previous_business_day'])))[0], format=MP_FXCM_DATETIME_FORMAT).date())

#%%
# delta=np.round(mean_range/50*100000)/100000
# max_date=max(candles_df.date)
# x=['#','%']
# # candles_df.alpha
# mask = np.logical_not(candles_df['alpha'].isin(x))
# previous_business_day = list(candles_df[candles_df.date==max_date].previous_business_day)[0]
# previous_df=candles_df.date==previous_business_day
# bool_list=(candles_df.date==previous_business_day)&mask
# index_list=[i for i, x in enumerate(bool_list) if x]
# df=candles_df.iloc[index_list]

# open_point = df[df.mp_period==14].open.values[0]
#%%
# Calculate the point of control for a particular day:
delta = delta_fn(candles_df=candles_df, increments=100, significant_figs=5)
d=candles_df_date_list[0]
df=candles_df[candles_df.date==d].copy()

lower_percentage=0.2
higher_percentage=0.8
def mp_analysis(df, lower_percentage, higher_percentage):
   delta_list=[]
   mp_dict={}
   # result_dict={}
   for mp_period in list(df.mp_period):
      # mp_period = list(df.mp_period)[0]
      # print(mp_period)
      index=index_fn(df, mp_period)
      low, high =high_low_fn(df,index)
      # mp_low, mp_high  = high_low_fn(df, index)
      mp_low, mp_high  = mp_high_low_fn(0, low, high, delta)
      alpha=alpha_fn(df, mp_period)   # print(low,high,mp_high,mp_low)
      # print(list(range(mp_low, mp_high)))
      # alpha_list=(mp_high-mp_low)*[df[index].alpha.values[0]]
      mp_dict = update_mp_dict(mp_dict, mp_low, mp_high, alpha)
      # print((mp_high-mp_low)*[df[index].alpha.values[0]])
      # print(list(range(mp_low, mp_high)))
      delta_list=delta_list+list(range(mp_low, mp_high))
   delta_list=list(set(delta_list))
   mp_dict=dict(sorted(mp_dict.items()))
   sum_print_list, poc_len = sum_print_fn(mp_dict)
   poc = poc_fn(mp_dict,  poc_len)
   totalprints=sum_print_list[-1]
   low_range, top_range = define_mp_range(totalprints, lower_percentage, higher_percentage)
   low_range_price, top_range_price = define_mp_pricerange(delta_list, sum_print_list, low_range, top_range, 5)
   d=10**4
   high=int(max(df.high)*d)/d
   low=int(min(df.low)*d)/d
   close=df[df.datetime==max(df.datetime)]['close'].values[0]
   openprice=df[df.datetime==min(df.datetime)]['open'].values[0]
   closeprice=int(close*d)/d
   openprice=int(openprice*d)/d
   # s=[mp_result_dict[datetime.date(2019, 12, 31)]['mp_dict'][p][-1] for p in poc]
   poc_alpha_list = [mp_dict[p][-1] for p in poc]
   # poc_alpha_set=list(set([mp_dict[p][-1] for p in poc]))
   # maxd = max([df[df.alpha==f]['datetime'].values[0] for f in result['poc_alpha']])
   poc_time_list = [df[df.alpha==f]['datetime'].values[0] for f in poc_alpha_list]
   # mp_dict
   m=max(poc_time_list)
   mp_poc = poc[poc_time_list.index(m)]
   return {'mp_dict':mp_dict,
           'mp_poc':mp_poc,
           'poc':poc,
           'npoc':len(poc),
           'poc_alpha_list':poc_alpha_list,
           'poc_time_list':poc_time_list,
           'poc_len':poc_len,
           'delta':delta,
           'total_deltas':len(sum_print_list),
           'low_range_price':low_range_price,
           'top_range_price':top_range_price,
           'high':high,
           'low':low,
           'closeprice':closeprice,
           'openprice':openprice}
#%%
lower_percentage=0.2
higher_percentage=0.8

def mp_result_dict_fn(candles_df, lower_percentage, higher_percentage):
   mp_result_dict={}
   candles_df_date_list=list(set(list(candles_df['date'])))
   candles_df_date_list.sort()
   for d in candles_df_date_list:
      # print(d)
      df=candles_df[candles_df.date==d].copy()
      result = mp_analysis(df, lower_percentage, higher_percentage)
      today=list(set(list(df.date)))[0]
      print('date = ',today, ', poc = ',result['poc'], ', poc_len = ',result['poc_len'], ', low_range_price = ',result['low_range_price'], ', top_range_price = ',result['top_range_price'])
      mp_result_dict[today]=result
   return mp_result_dict

mp_result_dict = mp_result_dict_fn(candles_df, lower_percentage, higher_percentage)
#%%
# Save and load mp_result_dict:

pname='/home/lnr-ai/github_repos/fxcm/data/'
with open(pname+'mp_result_dict.p', 'wb') as fp:
    pickle.dump(mp_result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(pname+'mp_result_dict.p', 'rb') as fp:
    mp_result_dict = pickle.load(fp)
#%%
def previous_business_day_poc_fn(mp_result_dict,today,prefbday):
   mp_poc=0
   unknownkey=False
   try:
      mp_poc=mp_result_dict[prefbday.date()]['mp_poc']
   except KeyError:
      unknownkey=True
      print("poc for {prefbday} is UNKNOWN".format(prefbday=prefbday))
   return mp_poc, unknownkey

def previous_business_day_range_fn(mp_result_dict,today,prefbday):
   low_range_price=0
   high_range_price=0   
   unknownkey=False
   try:
      low_range_price=mp_result_dict[prefbday.date()]['low_range_price']
      high_range_price=mp_result_dict[prefbday.date()]['top_range_price']
   except KeyError:
      unknownkey=True
      print("poc for {prefbday} is UNKNOWN".format(prefbday=prefbday))
   return low_range_price,high_range_price,unknownkey


def previous_business_day_fn(candles_df, today):
   df=candles_df[candles_df.date==today].copy()
   prefbday=df['previous_business_day'].values[0]
   return pd.to_datetime(prefbday)

def next_business_day_fn(candles_df, today):
   df=candles_df[candles_df.previous_business_day==today].copy()
   nextbday=df['date'].values[0]
   return pd.to_datetime(nextbday)

def OHLC_fn(candles_df, now):
   ohlc={}
   ohlc['datetime'] = now
   ohlc['date'] = candles_df.loc[candles_df['datetime']==now,'date'].values[0]
   ohlc['open'] = candles_df.loc[candles_df['datetime']==now,'open'].values[0]
   ohlc['high'] = candles_df.loc[candles_df['datetime']==now,'high'].values[0]
   ohlc['low'] = candles_df.loc[candles_df['datetime']==now,'low'].values[0]
   ohlc['close'] = candles_df.loc[candles_df['datetime']==now,'close'].values[0]
   return ohlc

def split_times_fn(now):
   return now - timedelta(hours=0, minutes=30), now + timedelta(hours=0, minutes=30)
   
#%%
today_list=[]
candles_df['previous_business_day_poc']=0
for today in mp_result_dict:
   print(today)
   today_list.append(today)
   print(mp_result_dict[today]['mp_poc'])
   prefbday=previous_business_day_fn(candles_df, today)
   # df=candles_df[candles_df.date==today].copy()
   # prefbday=df['previous_business_day'].values[0]
   # # candles_df[candles_df.date==prefbday]
   # # prefbday.astype(datetime)
   # s=pd.to_datetime(prefbday)
   previous_business_day_poc, unknownkey=previous_business_day_poc_fn(mp_result_dict,today,prefbday)
   # candles_df['previous_business_day_poc']=0
   if not unknownkey:
      candles_df.loc[candles_df.date==today, 'previous_business_day_poc']=previous_business_day_poc

now=candles_df.datetime[103]
now_prev, now_next = split_times_fn(now)
ohlc_now = OHLC_fn(candles_df, now)
ohlc_now_prev = OHLC_fn(candles_df, now_prev)
ohlc_now_next = OHLC_fn(candles_df, now_next)
mp_result_dict[ohlc_now['date']]


for dt in list(set(candles_df.date)):
   prefbday=previous_business_day_fn(candles_df, now)
   previous_business_day_poc=previous_business_day_poc_fn(mp_result_dict,now,prefbday)
   previous_business_day_range_fn(mp_result_dict,now,prefbday)   
   df=candles_df[candles_df.date==dt].copy()
   for now in df.datetime:
      now_prev, now_next = split_times_fn(now)
      ohlc_now = OHLC_fn(df, now)
      ohlc_now_prev = OHLC_fn(df, now_prev)
      ohlc_now_next = OHLC_fn(df, now_next)
      mp_result_dict[ohlc_now['date']]


for dt in candles_df.prev_datetime:
   try:
      print(dt,candles_df.loc[candles_df['datetime']==dt,'high'])
      candles_df.loc[candles_df['prev_datetime']==dt,'prev_high']=candles_df.loc[candles_df['datetime']==dt,'high'].values[0]
      candles_df.loc[candles_df['prev_datetime']==dt,'prev_low']=candles_df.loc[candles_df['datetime']==dt,'low'].values[0]
      
   except IndexError:
      # unknownkey=True
      print("previous high for {dt} is UNKNOWN".format(dt=dt))      
      
d = datetime.today() - timedelta(hours=0, minutes=50)
#%%

fig = plt.figure(figsize=(24, 18))
title_name='USDZAR'
fname=''
fig.suptitle('{fname}'.format(fname=title_name), fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
# ax.set_title('eurusd profile')
ax.set_xlabel('')
ax.set_ylabel('')
# mp.alpha_df['alpha_word_column'][0]
# mp.alpha_df['price_column'][0]
min_price=10**10
max_price=0
for price in mp_result_dict[today]['mp_dict']:
   print(int((price-prev_mp_poc)/delta))
   ax.text(0+0.05, price, mp_result_dict[today]['mp_dict'][price], horizontalalignment='left',verticalalignment='center',fontsize=10)
   min_price=min(min_price, price)
   max_price=max(max_price, price)

# ax.plot([2], [1], 'o')
# ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
#             arrowprops=dict(facecolor='black', shrink=0.05))
x = ["monday","tuesday","wednesday","thursday","friday"]
ax.axis([0, 5, min_price-0.01, max_price+0.01])
ax.set_xticklabels(x)
ax.grid(which='minor', axis='both', linestyle='--')
# ax.grid(color='r', linestyle='-', linewidth=1)
plt.grid()
plt.show()
fig.savefig('//home//charts//{gname}.png'.format(gname='EURUSD_profile'))
   
   
#%%
mp_result_dict[2019-12-11]
#%%
d=candles_df_date_list[0]
df=candles_df[candles_df.date==d].copy()
df=df.sort_values(by=['datetime'])
df.head(16)
mp_analysis(df.head(18), lower_percentage, higher_percentage)
#%%
delta_list=[]
mp_dict={}
d=candles_df_date_list[0]

df=candles_df[candles_df.date==d].copy()

for mp_period in list(df.mp_period):
   mp_period = list(df.mp_period)[0]
   # print(mp_period)
   index=index_fn(df, mp_period)
   low, high =high_low_fn(df,index)
   mp_low, mp_high  = mp_high_low_fn(open_point, low, high, delta)
   alpha=alpha_fn(df, mp_period)   # print(low,high,mp_high,mp_low)
   # print(list(range(mp_low, mp_high)))
   # alpha_list=(mp_high-mp_low)*[df[index].alpha.values[0]]
   mp_dict = update_mp_dict(mp_dict, mp_low, mp_high, alpha)
   print((mp_high-mp_low)*[df[index].alpha.values[0]])
   print(list(range(mp_low, mp_high)))
   delta_list=delta_list+list(range(mp_low, mp_high))

mp_dict_list=list(mp_dict)
mp_dict_list.sort()
mp_list=[mp_dict[x] for x in mp_dict_list]
mp_list_len=[len(x) for x in mp_list]
[i for i, x in enumerate([x==max(mp_list_len) for x in mp_list_len]) if x]
# [mp_dict_list[i] for i in p]
#%%

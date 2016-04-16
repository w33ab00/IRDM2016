######################################################
# Visualize Data
#
# 1. Load vs temperature (with respect to all 25 stations
# 2. Seasonality of the load
# 3. Differences in patterns during week and at weekend
# 4. Can we visualize holiday effect?
# 5. When we fit a simple perceptron, can we see outliers
# 6. Autocorrelation of the loads
######################################################

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as matdat
import datetime as dt
import pickle
import pdb
import copy

#############################################################################
# METHODS

def save_seasonality_plot(load_avg_hours, year):
    f = plt.figure(1)
    dat = matdat.date2num(load_avg_hours['dt'])
    plt.plot(load_avg_hours['dt'], load_avg_hours['LOAD'])
    plt.gcf().autofmt_xdate()
    plt.ylabel("energy load")
    title = "Seasonality of the energy load in %d" % year
    f.suptitle(title)
    fname = 'seasonality%d.png' % year
    f.savefig(fname)
    f.clear()


def save_weather_load_plot(dataframe, year, station):
    f1 = plt.figure(1)
    curstat = 'w' + station.__str__()
    plt.scatter(dataframe[curstat], dataframe['LOAD'], c=dataframe[curstat], cmap=plt.cm.afmhot)
    """
    unstable
    for station in range(1,5):
        plt.subplot(2,2,station)
        curstat = 'w' + station.__str__()
        plt.scatter(dataframe[curstat], dataframe['LOAD'], c=dataframe[curstat], cmap=plt.cm.afmhot)
    """
    title = "Weather and energy demand in %d" % year
    f1.suptitle(title)
    plt.xlabel("temperature")
    plt.ylabel("energy load")
    fname = 'loadvstemp%d.png' % year
    f1.savefig(fname)
    plt.close()

#############################################################################


try:
    df_train = pickle.load(open("df_train.p", "rb"))
    print "df_train loaded."

    df_2005 = df_train[df_train['year'] == 2005]
    df_2006 = df_train[df_train['year'] == 2006]
    df_2007 = df_train[df_train['year'] == 2007]
    df_2008 = df_train[df_train['year'] == 2008]
    df_2009 = df_train[df_train['year'] == 2009]
    df_2010 = df_train[df_train['year'] == 2010]

    forecasting_period = df_2010[(df_2010['month'] >=8) & (df_2010['day'] == 30)]
    forecasting_period.to_csv(path_or_buf="forecastingPeriod.csv")
    #pdb.set_trace()
    yr_dict = {
        2005: df_2005,
        2006: df_2006,
        2007: df_2007,
        2008: df_2008,
        2009: df_2009,
        2010: df_2010
    }

    # -----------------------------------------  1. Load vs temp (only for 2005)
    for yr in range(2005,2011):
        df = yr_dict[yr].groupby(['year', 'day'])['LOAD'].mean().reset_index()
        save_seasonality_plot(df, yr)
    # -----------------------------------------  2. Yearly pattern (only for 2005)
    # show how load varies over the year
    for yr in range(2005,2011):
        save_weather_load_plot(yr_dict[yr], yr, 1)
    # -----------------------------------------  3. Monthly pattern
    # mean over the same days in the dataframe
    # transform days from mday to wday, i.e. (1,..,31) -> (1,..,7)
    df_week_avg = pd.DataFrame({'day': [dd for dd in range(1,8)]})
    f = plt.figure(1)
    handles = []
    for yr in [2007, 2010]:
        df_tmp = yr_dict[yr]
        #df_tmp['day'] %= 7
        #df_tmp[df_tmp['day']==0] = 7
        df_tmp = yr_dict[yr].groupby(['year', 'day'])['LOAD'].mean().reset_index()
        yrstr = yr.__str__()
        plot = plt.plot(df_tmp['day'], df_tmp['LOAD'],label=yrstr)
        handles.append(plot[0])
    plt.legend(handles=handles, loc=3)
    plt.xlabel("day in month")
    plt.ylabel("energy load")
    f.savefig("avg_over_days_new.png")
    f.clear()
    # -----------------------------------------  3. Daily pattern
    handles = []
    for yr in range(2005, 2011):
        df_tmp = yr_dict[yr].groupby(['year', 'hour'])['LOAD'].mean().reset_index()
        yrstr = yr.__str__()
        plot = plt.plot(df_tmp['hour'], df_tmp['LOAD'], label=yrstr)
        handles.append(plot[0])
    plt.legend(handles=handles, loc=2)
    plt.xlabel("hour")
    plt.ylabel("energy load")
    f.savefig("avg_over_hours_new.png")
    f.clear()





except IOError:
    print "df_train.p not found. creating..."
    # LOAD DATASET
    df_train = pd.read_csv("/Users/mh/Documents/CSML/IRDM/GroupCW/GEFCom2014 Data/Load/Task 1/L1-train.csv",
                           delimiter=",", encoding="utf-8", header=0)
    # MANIPULATE DATASET
    lastIdxLoadMissing = df_train[df_train.LOAD.isnull()].index[-1]
    df_train = df_train[lastIdxLoadMissing+1:] # dataframe from beginning on valid load entry till end
    df_train = df_train.reset_index(drop='True') # reset the indexing to begin from zero
    # first replace those stupid time encodings
    df_train['year'] = 0
    df_train['month'] = 0
    df_train['day'] = 0
    df_train['hour'] = 0
    # populate 'year', 'month', 'day', 'hour' column of df_train
    for i in range(len(df_train.index)):
        if i % 1000:
            print "iteration %d" % i
        tt = df_train.loc[i, 'TIMESTAMP']
        tt_dt = dt.datetime.strptime(tt, "%m%d%Y %H:%M")
        tt_tuple = tt_dt.timetuple()
        df_train.loc[i, 'year'] = tt_tuple.tm_year
        df_train.loc[i, 'month'] = tt_tuple.tm_mon
        df_train.loc[i, 'day'] = tt_tuple.tm_mday
        df_train.loc[i, 'hour'] = tt_tuple.tm_hour
    # save df_train
    pickle.dump(df_train, open("df_train.p", "wb"))










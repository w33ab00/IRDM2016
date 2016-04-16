import time
import datetime as dt
import numpy as np
import pandas as pd
import math
import sys
from pandas.tseries.holiday import USFederalHolidayCalendar

# load the data into dataframe
df_train = pd.read_csv("../data/L1-train.csv", delimiter=",", encoding="utf-8", header=0)

# need to fix weird timestamps, no ability to disambiguate e.g. "1212005" 1st of month 12 and 21th of month 1 !!
cur_ts = dt.datetime.strptime('01012001 01:00', "%m%d%Y %H:%M")

correct_ts = [dt.datetime.strftime(cur_ts + dt.timedelta(hours=x), '%m%d%Y %H:%M') 
              for x in range(df_train.shape[0])]

# put the corrected timestamps back into the dataframe
df_train.loc[:,'TIMESTAMP'] = correct_ts

# initial data is mission load information, so need to get past that
lastIdxLoadMissing = df_train[df_train.LOAD.isnull()].index[-1]

df_train = df_train[lastIdxLoadMissing+1:] # datafram from beginning on valid load entry till end
df_train = df_train.reset_index(drop='True') # reset the indexing to begin from zero

# number of data-points
N_Data = np.shape(df_train)[0]

# Feature creation
# Per hour, we need 
#             - two inputs for the day of year (sin, cos)                                 (2)
#             - two inputs for the day of week (sin, cos)                                 (2)
#             - two inputs for the hour of day (sin, cos)                                 (2)
#             - temperature from every station from same hour                             (25)
#             - avg. temps of last 3 days                                                 (3)
#             - max, min tmp of last week                                                 (2)
#             - flag for holiday                                                          (1)
#             - Load history: energy load for same hour and same day the week before      (1)
#                             energy load from previous 24 hours                          (24)
# --> 62 features

# prepare feature vectors
N_Feat = 62

def circular(val, period):
    ret = [(math.sin(2*math.pi*x/period), math.cos(2*math.pi*x/period)) for x in [val]]
    return [ret[0][0], ret[0][1]]

def isHoliday(d):
    cal = USFederalHolidayCalendar()
    y_str = d.strftime("%Y")
    holidays = cal.holidays(start=y_str+'-01-01', end=y_str+'-12-31').to_pydatetime()
    if d in holidays:
        return 1
    else:
        return 0


    
# leave out 1st 30 days so that while making features, we can refer to past data    
N_leave_at_beginning = 400       
    
# the feature matrix
X_Data = np.zeros((N_Data, N_Feat))

# the outputs
Y_Data = np.zeros((N_Data,1))

# process all dataframe into features
for i in range(N_Data):
    
    # do not feature process the initial data in the series (since we look back)
    if i < N_leave_at_beginning:
        continue;
    
    # the outcome
    Y_Data[i] = df_train.ix[i].LOAD
    
    tstamp = df_train.ix[i].TIMESTAMP
    curdt = dt.datetime.strptime(tstamp, "%m%d%Y %H:%M")
    ttpl = curdt.timetuple()
    doy = ttpl.tm_yday # day of year
    dow = ttpl.tm_wday # day of week
    hod = ttpl.tm_hour # hour of the day
    
    doyfeat = circular(doy, 365)
    dowfeat = circular(dow, 7)
    hodfeat = circular(hod, 24)
    
    X_Data[i][0] = doyfeat[0]
    X_Data[i][1] = doyfeat[1]
    X_Data[i][2] = dowfeat[0]
    X_Data[i][3] = dowfeat[1]
    X_Data[i][4] = hodfeat[0]
    X_Data[i][5] = hodfeat[1]
    
    
    # temperateure at 25 stations at the current hour  #######################################
    tempdata = df_train.ix[i].values[3:] 
    
    for idx in range(6,6+25):
        X_Data[i][idx] = tempdata[idx-6]
    
    
    # for all previous calcs, use the window
    
    df_Window = df_train.ix[(i-400):i]
    
    # avg. temps of last 3 days  ##############################################################
    prev_dates = [curdt - dt.timedelta(days=gap) for gap in [1,2,3]] 
    
    av_temps = []  
    for j in range(3):
        prevdate = prev_dates[j]
        prevdate_str = dt.datetime.strftime(prevdate, "%m%d%Y %H:%M")
        df_prevdate = df_Window[df_Window.TIMESTAMP.str.contains('^'+ prevdate_str)]
        if df_prevdate.shape[0] > 0:
            # get average temperature 
            temps = df_prevdate[['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 
                   'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 
                   'w20', 'w21', 'w22', 'w23', 'w24', 'w25']].values
            av_temps.append(np.mean(temps));
        else:
            print "Error"
            sys.exit()
                    
    
    X_Data[i][31] = av_temps[0]
    X_Data[i][32] = av_temps[1]
    X_Data[i][33] = av_temps[2]
    
    # max, min temp of last week  ##############################################################
    # - week is monday to friday plus the following sat + sun
    # get all days of previous week
    minmaxtemps = []
    
    prev_monday = curdt - dt.timedelta(days=curdt.weekday()) + dt.timedelta(days=0, weeks=-1) # day 0 is monday
    prev_dates = [prev_monday + dt.timedelta(days=x) for x in range(7)]
    
    dftemp = pd.DataFrame()
    for j in range(7):
        prevdate = prev_dates[j]
        prevdate_str = dt.datetime.strftime(prevdate, "%m%d%Y %H:%M")
        curdf = df_Window[df_Window.TIMESTAMP.str.contains('^'+ prevdate_str)]
        dftemp = dftemp.append(curdf);
    
    # get min and max temp
    temps = df_prevdate[['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 
               'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 
               'w20', 'w21', 'w22', 'w23', 'w24', 'w25']].values
    minmaxtemps = [np.min(temps), np.max(temps)]
        
    X_Data[i][34] = minmaxtemps[0]
    X_Data[i][35] = minmaxtemps[1]
    
    # holiday?
    holiday = isHoliday(curdt)
    X_Data[i][36] = holiday
    
    # load (same hour same day, the week before) #########################################
    prevLoad = 0.0
    sameHourWeekBefore = curdt + dt.timedelta(weeks=-1)
    prevdate_str = dt.datetime.strftime(sameHourWeekBefore, "%m%d%Y %H:%M")

    df_prevdate = df_Window[df_Window.TIMESTAMP.str.contains('^'+ prevdate_str)]
    if df_prevdate.shape[0] > 0:
        prevLoad = df_prevdate.values[0][2]  # the load value
        
    X_Data[i][37] = prevLoad
    
    # energy load prev 24 hours  #########################################
    prevHours = [curdt + dt.timedelta(hours=-(x+1)) for x in range(24)]
    prevdates_str = [dt.datetime.strftime(x, "%m%d%Y %H:%M") for x in prevHours]
    prevLoads = [df_Window[df_Window.TIMESTAMP == x].LOAD.values[0] for x in prevdates_str]

    for idx in range(38,38+24):
        X_Data[i][idx] = prevLoads[idx-38]
        
    
    if i % 100 == 0:
        print i, "of ", N_Data
    
# Now the feature vectors and output vectors are ready
print "Done making X_Data, Y_Data"

# Save them
np.save('../data/X_Data.npy', X_Data)
np.save('../data/Y_Data.npy', Y_Data)

# save same data but with 24-hour load history beginning 48/72/98 hours before:
print "saving features with shifted time windows...."

# go inside X_Data, where non-zero entries begin, and process df_train from that index onward
idxNzRow = [idx for idx, v in enumerate(~(X_Data==0).all(1)) if v == True][0]

load_idx_begin = 61-24+1   # last 24 cols in Data are load history

for slidebackto in [24, 48, 72, 98]:
    for idx in range(idxNzRow, X_Data.shape[0]):
        idxbegin = (idx - slidebackto)
        prevData = df_train.ix[idxbegin:(idxbegin + 24 - 1)]
        X_Data[idx, load_idx_begin:(load_idx_begin + 24)] = prevData.LOAD.values
        if idx % 5000 == 0:
            print "processed ", idx

    fname = "../data/X_Data_mod_" + str(slidebackto) + ".npy"
    np.save(fname, X_Data)
    print "wrote " + fname
print "All done"  

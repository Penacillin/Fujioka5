from scraper import getBarData, getBarDataWithTimestamp, dataDir
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
LOG_DATA = False # Config to log10() each data sample

def convertHLCBarsToHLCDiffs(HLCBarData):
    """
    Converts HLCBarData to Difference Bar Data
    """
    #print(HLCBarData)
    diffData = None
    if LOG_DATA:
        diffData = np.log(HLCBarData)
        diffData = np.diff(diffData, axis=0)
    else:
        diffData = np.diff(HLCBarData, axis=0)
       
    #print(diffData)
    return diffData

def convertHLCBarsToHLCStandadNorm(HLCBarData):
    """r
    Converts HLCBarData to standard normalization bars
    """
    #print(HLCBarData)
    # diffData = None
    # if LOG_DATA:
    #     diffData = np.log(HLCBarData)
    #     diffData = np.diff(diffData, axis=0)
    # else:
    #     diffData = np.diff(HLCBarData, axis=0)
    print(HLCBarData.shape)
    stdData = np.std(HLCBarData.astype(np.float64), axis=0)
    meanData = np.mean(HLCBarData.astype(np.float64), axis=0)
    print(stdData.shape)
    print(meanData.shape)
    HLCBarData = (HLCBarData - meanData) / stdData
    print(HLCBarData)
    return HLCBarData

def getPDDataDiffed(dateFrom, dataRange, timeFrame, pair):
    '''
    Get data from dateFrom-dataRange to dateFrom with timeframe for pair
    as a dataFrame with HLC DIFFS and the raw bar data itself

    Return: 
    pdData : pd.DataFrame
    barData : np.ndarray
    '''
    # Get data
    startTime = dateFrom - dataRange
    print("getting data from {} to {}".format(startTime, dateFrom))
    barData = loadBars(startTime, dateFrom, pair, timeFrame)
    if barData is None:
        barData = getBarData(startTime, dateFrom, pair, getTimeFrame(timeFrame))
    data = convertHLCBarsToHLCDiffs(barData[:,[1,2,3]])
    pdData = pd.DataFrame(data, index= (barData[1:,0]), columns=['high', 'low', 'close'])
    #print(pdData[startTime:dateFrom])
    #print()
    return pdData[startTime:dateFrom], barData

def getPDData(dateFrom, dataRange, timeFrame, pair):
    '''
    Get data from dateFrom-dataRange to dateFrom with timeframe for pair
    as a dataFrame with HLC DIFFS and the raw bar data itself

    Return: 
    pdData : pd.DataFrame
    barData : np.ndarray
    '''
    # Get data
    startTime = dateFrom - dataRange
    print("getting data from {} to {}".format(startTime, dateFrom))
    barData = loadBars(startTime, dateFrom, pair, timeFrame)
    if barData is None:
        barData = getBarData(startTime, dateFrom, pair, getTimeFrame(timeFrame))
    #data = convertHLCBarsToHLCStandadNorm(barData[:,[1,2,3]])
    pdData = pd.DataFrame(barData[:,[1,2,3]], index= (barData[:,0]), columns=['high', 'low', 'close'])
    #print(pdData[startTime:dateFrom])
    #print()
    return pdData[startTime:dateFrom], barData


def getTimeFrame(timeFrame):
    if timeFrame == "H1":
        return timedelta(0,0,0,0,0,1,0)
    elif timeFrame == "H4":
        return timedelta(0,0,0,0,0,4,0)
        

def compileTickDataToBars(startDate, endDate, pair, timeFrame):
    barData = getBarDataWithTimestamp(startDate, endDate, pair, getTimeFrame(timeFrame))
    print(barData.shape)
    print(len(barData))
    filePath = dataDir + pair + timeFrame
    print("Saving array to:", filePath)
    np.save(filePath, barData)
    print("Finished saving")

def loadBars(startDate, endDate, pair, timeFrame):
    """
    Load bars from saved npy file from startDate to endDate
    """
    filePath = dataDir + pair + timeFrame + '.npy'
    print("Loading from filepath", filePath)
    if os.path.isfile(filePath):
        data = np.load(filePath, allow_pickle=True)
        print(data)
        return data
    else:
        print("FAILED TO LOAD BARS from numpy")
        return None


def testBars():
    barData = getBarData(datetime(2018,1,4), datetime(2018,1,10), "AUDUSD", getTimeFrame("H1"))
    
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go

    open_data = barData[:,2]
    open_data = np.insert(open_data, 0, 0)
    high_data = barData[:,0]
    low_data = barData[:,1]
    close_data = barData[:,2]
    dates = [i for i in range(0,len(close_data))]

    trace = go.Ohlc(x=dates,
                    open=open_data,
                    high=high_data,
                    low=low_data,
                    close=close_data)
    data = [trace]
    plotly.offline.plot(data, filename='ohlc_datetime.html')

    diffHLCData = convertHLCBarsToHLCDiffs(barData)

    import plotly.tools as tls
    import matplotlib.pyplot as plt

    print("DIFF DATA:")
    y = diffHLCData[:,0]
    x = range(len(y))
    # Create a trace
    trace2 = go.Histogram(
        x = y
    )

    data2 = [trace2]

    # Plot and embed in ipython notebook!
    plotly.offline.plot(data2, filename='basic-scatter.html')


def main():
    compileTickDataToBars(datetime(2010, 1, 1), datetime(2018,10,30), "AUDUSD", "H1")

    exit(0)

if __name__ == "__main__":
    main()

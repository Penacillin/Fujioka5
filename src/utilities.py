import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

def plotBarData(barData, filename):
    """
    Plots HLC data from barData


    """
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
    plotly.offline.plot(data, filename='../plots/'+filename)

def plotHistogram(barData, filename):
    data = [go.Histogram(x=barData[:,2])]
    plotly.offline.plot(data, filename='../plots/'+filename)
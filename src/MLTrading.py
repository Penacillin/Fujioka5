﻿from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import argparse
from multiprocessing import Process
from math import sqrt

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn import linear_model

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments.               #
# Users could set them from the project setting page.             #
###################################################################

input_dir = None
output_dir = None
log_dir = None


#################################################################################
# Keras configs.                                                                #
# Please refer to https://keras.io/backend .                                    #
#################################################################################

import tensorflow as tf
from tensorflow.keras import layers


#K.set_floatx('float32')
#String: 'float16', 'float32', or 'float64'.

#K.set_epsilon(1e-05)
#float. Sets the value of the fuzz factor used in numeric expressions.


#################################################################################
# Keras imports.                                                                #
#################################################################################

from preprocess import convertHLCBarsToHLCDiffs,  getPDDataDiffed, getPDData
from scraper import getBarData
from utilities import plotBarData, plotHistogram

from datetime import datetime, timedelta

import plotly.plotly as py
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR

def calculateRMSE(data, preds):
    """
    Given 2 equally shaped ndarrays, does elementwise Root mean squared error
    """
    return sqrt(np.mean(np.square(data - preds)))

info_criterias = ['aic', 'fpe', 'hqic', 'bic']
fit_ic = 'fpe'

def VARPredict(pdData, steps : int):
    """
    Takes a pandas dataframe, then predicts steps ahead

    pdData: Pandas
    steps: int

    Returns a (steps, series) shaped ndarray of forecast
    """
    # Compute VAR
    model = VAR(pdData)
    
    # fit data
    try:
        results = model.fit(maxlags=15, ic=fit_ic, trend='nc')
    except Exception as e:
        print(e)
        print(pdData)
        exit(1)
    #print(results.summary())

    #results.plot()
    #plt.show()
    #results.plot_acorr()
    #plt.show()

    #Forecast diffs
    forecast = results.forecast(pdData.values[-results.k_ar:], steps)
    #results.plot_forecast(10)
    #plt.show()

    return forecast     





def continuousVARPredict(pdData, barHistory):
    """
    Rolls through pdData , and uses up to barHistory worth of historical bars to predict
    next bar

    Returns a (len(pdData)-barHistory+1, steps, series) shaped ndarray of forecast
    """

    takeIndices = [i for i in range(barHistory)]
    forecastedDiffs = []
    for i in range(0, len(pdData)-barHistory+1):
        #print(takeIndices)
        currData = np.take(pdData, takeIndices, 0)
        currData = currData.reset_index()
        currData = currData.drop(columns="index")
        currData = currData.astype(float)
        # append prediction
        forecastDiffs = VARPredict(currData, 1)
        forecastedDiffs.append(forecastDiffs)
        # Increment window
        takeIndices = np.add(takeIndices, 1)

    print("With {0} bars of data, predicted {1} data points".format(len(pdData), len(forecastedDiffs)))
    return np.asarray(forecastedDiffs)


def continuousLRidgePredict(pdData, barHistory):
    takeIndices = [i for i in range(barHistory)]
    forecastedDiffs = []
    for i in range(0, len(pdData)-barHistory+1):
        #print(takeIndices)
        currData = np.take(pdData, takeIndices, 0)
        currData = currData.reset_index()
        currData = currData.drop(columns="index")
        currData = currData.astype(float)
        # append prediction
        forecastDiffs = VARPredict(currData, 1)
        forecastedDiffs.append(forecastDiffs)
        # Increment window
        takeIndices = np.add(takeIndices, 1)

    print("With {0} bars of data, predicted {1} data points".format(len(pdData), len(forecastedDiffs)))
    return np.asarray(forecastedDiffs)

STATE_DIM = 3
ACTION_DIM = 3

def get_trader_model(state_in):
    # TODO: Define Network Graph
    layer1_w = tf.layers.Dense(1000,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=5),
                bias_initializer=tf.random_normal_initializer(mean=0.,stddev=5),
                name="mylayer")
    layer1 = layer1_w.apply(state_in)

    with tf.variable_scope("mylayer", reuse=True):
        weights = tf.get_variable("kernel")     

    layer2_w = tf.layers.Dense(ACTION_DIM,
                activation=None, use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=0.18/np.sqrt(1000)))
    layer2 = layer2_w.apply(layer1)

    # TODO: Network outputs
    q_values = layer2
    return q_values
    # return model

def run_trainer():
    

    state_in = tf.placeholder("float", [None, STATE_DIM])
    action_in = tf.placeholder("float", [None, ACTION_DIM])
    target_in = tf.placeholder("float", [None])

BATCH_SIZE = 64
WINDOW_SIZE = 60

def get_soft_predictor_model():
    inputs = layers.Input(shape=(WINDOW_SIZE,3), batch_size=BATCH_SIZE,
                            dtype=tf.float32,name='input_layer')
    x = layers.Flatten()(inputs)

    x = layers.Dense(500, activation='relu', dtype=tf.float32,name='500_dense_layer')(x)
    x = layers.Dropout(rate=0.5, dtype=tf.float32, name='dropout_layer_1')(x)
    x = layers.Dense(200, activation='relu', dtype=tf.float32, name='200_dense_layer')(x)
    x = layers.Dropout(rate=0.5, dtype=tf.float32, name='dropout_layer_2')(x)
    x = layers.Dense(40, activation='relu', dtype=tf.float32)(x)
    x = layers.Dense(20, activation='relu', dtype=tf.float32)(x)
    # trend_output = layers.Dense(3, activation='softmax', dtype=tf.float32)(x)
    value_output = layers.Dense(3, dtype=tf.float32)(x)

    model = tf.keras.Model(inputs=inputs, outputs=[value_output])


    return model

def get_meme_model():
    inputs = layers.Input(shape=(WINDOW_SIZE,3), batch_size=BATCH_SIZE,
                            dtype=tf.float32,name='input_layer')
    # value_output = tf.slice(inputs, [0, WINDOW_SIZE-1,0], [BATCH_SIZE,WINDOW_SIZE-1,3])
    x = layers.Lambda(lambda x: tf.slice(x, [0, WINDOW_SIZE-1,0], [BATCH_SIZE,1,3]))(inputs)
    value_output = layers.Flatten()(x)
    model = tf.keras.Model(inputs=inputs, outputs=[value_output])


    return model

def main():    
    global info_criterias
    global fit_ic
    endTime =   datetime(2017,9,15,14)
    timeFrame = timedelta(0,0,0,0,0,1,0)
    timeRange = timedelta(400)
    # Get data
    pdData, barData = getPDData(endTime, timeRange, "H1", "AUDUSD")

    print("got " + str(pdData.shape) + " of data" + " for range " + str(timeRange.days*24))
    
    # Split data into training and testing
    raw_training_size = int(len(pdData)*0.8)
    raw_testing_size = len(pdData) - raw_training_size
    raw_training_data = pdData.iloc[:raw_training_size]
    raw_testing_data = pdData.iloc[raw_training_size:]
    
    print("raw training data:", raw_training_data.shape, len(raw_training_data), raw_training_size)
    print("raw testing data:", raw_testing_data.shape, len(raw_testing_data))


    # Normalize data
    # training data: (samples-window, 3*window)   
    #  training labels: (samples-window, 3)
    training_data = []
    training_trend_labels = []
    training_value_labels = []
    i = 0
    for s in range((len(raw_training_data)-WINDOW_SIZE)-(len(raw_training_data)-WINDOW_SIZE)%BATCH_SIZE):
        sample = raw_training_data.iloc[s:s+WINDOW_SIZE].to_numpy()
        label = raw_training_data.iloc[s+WINDOW_SIZE].to_numpy()
        sample_mean = np.mean(sample)
        sample_stddev = np.std(sample)
        sample_normalized = (sample-sample_mean)/sample_stddev
        label_normalized = (label-sample_mean)/sample_stddev

        training_data.append(sample_normalized)
        training_value_labels.append(label_normalized)


    training_data = np.array(training_data)
    training_value_labels = np.array(training_value_labels)


    testing_data = []
    testing_trend_labels = []
    testing_value_labels = []
    for s in range((len(raw_testing_data)-WINDOW_SIZE)-(len(raw_testing_data)-WINDOW_SIZE)%BATCH_SIZE):
        sample = raw_testing_data.iloc[s:s+WINDOW_SIZE].to_numpy()
        label = raw_testing_data.iloc[s+WINDOW_SIZE].to_numpy()
        sample_mean = np.mean(sample)
        sample_stddev = np.std(sample)
        sample_normalized = (sample-sample_mean)/sample_stddev
        label_normalized = (label-sample_mean)/sample_stddev

        testing_data.append(sample_normalized)
        testing_value_labels.append(label_normalized)


    testing_data = np.array(testing_data)
    testing_value_labels = np.array(testing_value_labels)
    
    print(training_data.shape)
    print(training_value_labels.shape)
    print(testing_data.shape)
    print(testing_value_labels.shape)

    # Training
    meme_model = get_meme_model()
    meme_model.compile(
        optimizer='adam',
        loss=['mean_squared_error'],
        loss_weights=[1.],
        metrics=['accuracy','mean_absolute_error', 'mean_squared_error']
    )

    meme_model.summary()
    meme_model.fit(training_data, training_value_labels, batch_size=BATCH_SIZE, epochs=5)

    model = get_soft_predictor_model()
    model.compile(
        optimizer='adam',
        loss=['mean_squared_error'],
        loss_weights=[1.],
        metrics=['accuracy','mean_absolute_error', 'mean_squared_error']
    )

    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model.fit(training_data, training_value_labels, batch_size=BATCH_SIZE, epochs=15, 
                callbacks=[early_stop])


    # EVALUATE
    print("\n\nEVALUATION:")
    loss, acc, mae, mse = model.evaluate(testing_data,testing_value_labels)
    print("{} {} {} {}".format(loss,acc,mae,mse))

    loss, acc, mae, mse = meme_model.evaluate(testing_data,testing_value_labels)
    print("{} {} {} {}".format(loss,acc,mae,mse))
    
    exit(0)

    # set parameters
    fit_ic = 'bic'
    print(">>>>>>>>>>>>>>>>>>>> Window size for prediction (bars):", window)
    # Get var forecasted diffs
    allForecastedDiffs = continuousVARPredict(raw_testing_size, window)
    allForecastedDiffs = allForecastedDiffs[:-1].reshape((len(allForecastedDiffs[:-1]) ,3))
    # Get baseline prediction diffs
    baselineForecastDiffs = np.zeros(allForecastedDiffs.shape)
    # Get lridge forecasted diffs
    reg = linear_model.Ridge(alpha=.5)
    
    reg.fit()
    # Get real differences
    realDiffs = pdDiffDataTruncated[window:].values
    # Compare forecasted diffs to actual diffs
    modelError = calculateRMSE(allForecastedDiffs, realDiffs)
    baselineError = calculateRMSE(baselineForecastDiffs, realDiffs)

    print(modelError)
    print(baselineError)
    print(modelError / baselineError)
    print(">>>>>>>>>>>>>>>>>>>>")

    forecastBaselineDiff = baselineForecastDiffs - allForecastedDiffs

    trace0 = go.Scatter(
                x=allForecastedDiffs[:,2],
                y=realDiffs[:,2]
            )
    trace1 = go.Scatter(
            x=baselineForecastDiffs[:,2],
            y=realDiffs[:,2]
        )
    data=[trace0, trace1]
    plotly.offline.plot(data, filename="pls.html")
        
    exit(0)

    # Forecast diffs
    forecastDiffs = VARPredict(pdDiffData, 1)
    print(forecastDiffs)
    exit(1)
    # Convert forecasted diffs to actual data
    forecastBars = np.concatenate([[barData[-1]], forecastDiffs], axis=0)
    forecastBars = np.cumsum(forecastBars,0).round(5)
    completeData = np.concatenate([barData[0:-1], forecastBars],0)
    
    # plot data
    plotBarData(barData, 'pre.html')
    plotBarData(completeData, 'forecast.html')
    
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, 
                        default=None, 
                        help="Input directory where where training dataset and meta data are saved", 
                        required=False
                        )
    parser.add_argument("--output_dir", type=str, 
                        default=None, 
                        help="Input directory where where logs and models are saved", 
                        required=False
                        )

    args, unknown = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    log_dir = output_dir

    main()

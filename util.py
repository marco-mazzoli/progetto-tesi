import pandas as pd
import numpy as np
from datetime import date
import csv 
import requests
import glob
from pandas import DataFrame
from pandas import concat

def fetch_csv_data(url, date):
    """
    This function fetches from CSV data from an URL and a date formatted as YYYYMMDD.
    The URL must contain the word PLACEHOLDER, where the date will be put.
    """
    formatted_url = url.replace('PLACEHOLDER', date)
    result = pd.read_csv(formatted_url)
    return result

def read_multiple_csv(path, regex=''):
    """
    Given a global path retruns a dataframe made of the csv present in that path
	with the ending of the files name controlled by regex
    """
    all_files = glob.glob(path + '/*' + regex + '.csv')
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame

def select_attributes(frame, attributes):
    return frame[attributes]

def select_relevant_rows(frame, row, filter):
    return frame[frame[row] == filter]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def read_movement_data(path, regex, region='', province=''):
    df = read_multiple_csv(path, regex)
    filter_region = np.NaN if region == '' else region
    filter_province = np.NaN if province == '' else province

    if region == '':
        df = df[df['sub_region_1'].isna()]
    else:
        df = df[df['sub_region_1'] == region]
    
    if province == '':
        df = df[df['sub_region_2'].isna()]
    else:
        df = df[df['sub_region_2'] == province]

    return df

# # walk-forward validation for univariate data
# def walk_forward_validation(data, n_test):
# 	predictions = list()
# 	# split dataset
# 	train, test = train_test_split(data, n_test)
# 	# seed history with training dataset
# 	history = [x for x in train]
# 	# step over each time-step in the test set
# 	for i in range(len(test)):
# 		# split test row into input and output columns
# 		testX, testy = test[i, :-1], test[i, -1]
# 		# fit model on history and make a prediction
# 		yhat = xgboost_forecast(history, testX)
# 		# store forecast in list of predictions
# 		predictions.append(yhat)
# 		# add actual observation to history for the next loop
# 		history.append(test[i])
# 		# summarize progress
# 		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
# 	# estimate prediction error
# 	error = mean_absolute_error(test[:, -1], predictions)
# 	return error, test[:, 1], predictions

# # split a univariate dataset into train/test sets
# def train_test_split(data, n_test):
# 	return data[:-n_test, :], data[-n_test:, :]

# # fit an xgboost model and make a one step prediction
# def xgboost_forecast(train, testX):
# 	# transform list into array
# 	train = asarray(train)
# 	# split into input and output columns
# 	trainX, trainy = train[:, :-1], train[:, -1]
# 	# fit model
# 	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
# 	model.fit(trainX, trainy)
# 	# make a one-step prediction
# 	yhat = model.predict([testX])
# 	return yhat[0]
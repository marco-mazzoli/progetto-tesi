import pandas as pd
import numpy as np
from datetime import date
import csv 
import requests
import glob
from pandas import DataFrame, read_csv
from pandas import concat
import requests
import os.path, time

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

def read_movement_data(path,regex='',region='',province=''):
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

def check_data_update_requirement(file_path):
    result = False
    if os.path.exists(file_path):
        created_time = time.strftime('%Y-%m-%d', time.gmtime(os.path.getmtime(file_path)))
        today = date.today()
        result = False if created_time != today else True
    else:
        result = True
    return result

def download_updated_mobility_data(mobility_data_url, file_path):
    if check_data_update_requirement(file_path):
        request = requests.get(mobility_data_url, allow_redirects=True)
        if request.ok:
            print('Success!')
            open(file_path, 'wb').write(request.content)
        else:
            print('Error while downloading file...')
        os.system('rm -rf ' + region_path)
        request = requests.get(mobility_data_zip_url, allow_redirects=True)
        if request.ok:
            print('Success!')
            open(zip_path, 'wb').write(request.content)
            os.system('unzip ' + zip_path + ' -d ' + region_path)
            os.system('unlink ' + zip_path)
        else:
            print('Error while downloading file...')
    else:
        ('Mobility data up to date...')
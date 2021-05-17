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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def fetch_csv_data(url,date):
    """
    This function fetches from CSV data from an URL and a date formatted as YYYYMMDD.
    The URL must contain the word PLACEHOLDER, where the date will be put.
    """
    formatted_url = url.replace('PLACEHOLDER', date)
    result = pd.read_csv(formatted_url)
    return result

def read_multiple_csv(path,regex=''):
    """
    Given a global path retruns a dataframe made of the csv present in that path
	with the ending of the files name controlled by regex
    """
    all_files = glob.glob(path + '/*' + regex + '.csv')
    li = []

    for filename in all_files:
        df = pd.read_csv(filename,index_col=None,header=0)
        li.append(df)

    frame = pd.concat(li,axis=0,ignore_index=True)
    return frame

def select_attributes(frame,attributes):
    return frame[attributes]

def select_relevant_rows(frame,row,filter):
    return frame[frame[row] == filter]

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
    if result == False:
        print('Data already up to date...')
    return result

def download_updated_mobility_data(mobility_data_url,file_path,region_path,mobility_data_zip_url,zip_path):
    if check_data_update_requirement(file_path):
        request = requests.get(mobility_data_url,allow_redirects=True)
        if request.ok:
            print('Success!')
            open(file_path, 'wb').write(request.content)
        else:
            print('Error while downloading file...')
        os.system('rm -rf ' + region_path)
        request = requests.get(mobility_data_zip_url,allow_redirects=True)
        if request.ok:
            print('Success!')
            open(zip_path, 'wb').write(request.content)
            os.system('unzip ' + zip_path + ' -d ' + region_path)
            os.system('unlink ' + zip_path)
        else:
            print('Error while downloading file...')
    else:
        ('Mobility data up to date...')

def train_and_predict(dataset,column_to_predict,n_days,n_predictions):
    dataset_reduced = dataset.iloc[-n_days:,:]

    y = dataset_reduced[column_to_predict]
    X = np.ascontiguousarray(dataset_reduced)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=n_predictions
        )
    regressor = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000
        )

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return y_pred, y_test

def select_optimal_window_linear(
    dataset,
    column_to_predict,
    max_predictions=3,
    min_pred=1,
    pred_step=1,
    min_days=250,
    days_step=75
    ):
    result = pd.DataFrame(columns=[
        'mae',
        'prediction_window',
        'train_window'
        ])
    for n_predictions in range(min_pred,max_predictions,pred_step):
        size = len(dataset) - n_predictions
        for n_days in range(min_days,size,days_step):
            y_pred,y_test = train_and_predict(
                dataset,
                column_to_predict,
                n_days,
                n_predictions
                )
            mae = mean_absolute_error(y_test,y_pred)
            current_result = {
                'mae':mae,
                'prediction_window':n_predictions,
                'train_window':n_days
                }
            result = result.append(current_result,ignore_index=True)
    return result
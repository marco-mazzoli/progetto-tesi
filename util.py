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
        created_time = time.strftime(
            '%Y-%m-%d',
            time.gmtime(os.path.getmtime(file_path))
            )
        today = date.today()
        result = created_time != str(today)
    else:
        result = True
    if result == False:
        print('Data already up to date...')
    return result

def download_updated_mobility_data(
    mobility_data_url,
    file_path,
    region_path,
    mobility_data_zip_url,
    zip_path
    ):
    if check_data_update_requirement(file_path):
        print('Downloading ' + mobility_data_url)
        request = requests.get(mobility_data_url,allow_redirects=True)
        if request.ok:
            print(file_path + ' downloaded!')
            open(file_path, 'wb').write(request.content)
        else:
            print('Error while downloading ' + mobility_data_url)
        os.system('rm -rf ' + region_path)
        print('Downloading ' + mobility_data_zip_url)
        request = requests.get(mobility_data_zip_url,allow_redirects=True)
        if request.ok:
            print(zip_path + ' downloaded!')
            open(zip_path, 'wb').write(request.content)
            os.system('unzip ' + zip_path + ' -d ' + region_path)
            os.system('unlink ' + zip_path)
        else:
            print('Error while downloading ' + mobility_data_zip_url)
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

def time_series_cross_validation(
    dataset,
    column_to_predict,
    max_predictions=10,
    min_pred=1,
    pred_step=3,
    min_days=100,
    days_step=75
    ):
    result = pd.DataFrame(columns=[
        'mae',
        'prediction_window',
        'train_window',
        'y_test',
        'y_pred'
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
                'train_window':n_days,
                'y_test':y_test,
                'y_pred':y_pred
                }
            result = result.append(current_result,ignore_index=True)
    return result

def select_time_slot(df,start_date,end_date):
    return df.loc[start_date:end_date]

def grangers_causation_matrix(data,variables,test='ssr_chi2test',verbose=False,maxlag=12):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables),len(variables))),columns=variables,index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

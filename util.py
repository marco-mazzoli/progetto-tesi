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
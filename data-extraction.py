# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from datetime import date
import csv 
import requests


# %%
today = date.today()
today = str(today).replace('-', '') #remove hyphen for URL formatting
print(today)


# %%
url_regional_data = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-PLACEHOLDER.csv'


# %%
def fetch_csv_data(url, date):
    """
    This function fetches from CSV data from an URL and a date formatted as YYYYMMDD.
    The URL must contain the word PLACEHOLDER, where the date will be put.
    """
    formatted_url = url.replace('PLACEHOLDER', date)
    result = pd.read_csv(formatted_url)
    return result


# %%
df = fetch_csv_data(url_regional_data, '20210102')


# %%
df.head()


# %%




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
import os.path


def data_loader():
    company_name = 'AAPL'
    a = os.path.isfile('apple.csv')
    if a is False:
        start = datetime.datetime(2010, 1, 1)
        end = datetime.datetime(2019, 1, 1)
        df_aapl1 = web.DataReader(company_name, 'yahoo', start, end)
        df_aapl1.to_csv(r'apple.csv', sep='\t', encoding='utf-8', header='true')
        df_aapl1 = pd.read_csv('apple.csv', sep='\t')
    else:
        df_aapl1 = pd.read_csv('apple.csv', sep='\t')
    # print(df_aapl)

    company_name = 'GOOG'
    a = os.path.isfile('google.csv')
    if a is False:
        start = datetime.datetime(2010, 1, 1)
        end = datetime.datetime(2019, 1, 1)
        df_goog1 = web.DataReader(company_name, 'yahoo', start, end)
        df_goog1.to_csv(r'google.csv', sep='\t', encoding='utf-8', header='true')
        df_goog1 = pd.read_csv('google.csv', sep='\t')
    else:
        df_goog1 = pd.read_csv('google.csv', sep='\t')
    # print(df_goog)
    return df_aapl1, df_goog1


''' Load data '''
df_apple, df_google = data_loader()

''' Merge values '''
df = pd.merge(df_apple, df_google, how='inner', left_index=True, right_index=True)

''' Plot '''
data_apple = df['Close_x'].to_numpy()
data_google = df['Close_y'].to_numpy()

plt.plot(np.arange(df_apple.shape[0]), data_apple, label='Apple')
plt.plot(np.arange(df_apple.shape[0]), data_google, label='Google')
plt.xlabel('Days')
plt.ylabel('Stock price ($)')
plt.title('Google vs. Apple Stock market price')
plt.legend()

plt.show()

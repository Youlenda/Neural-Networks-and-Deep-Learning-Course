import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import os.path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from sklearn.model_selection import train_test_split


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


def data_for_training(data_s1, period1):
    a1 = np.arange(len(data_s1) - period1 + 1)
    data_list1 = []
    for i in range(len(a1)):
        sample_list = []
        for j in range(period1):
            value1 = data_s1[i + j]
            sample_list.append(value1)
        data_list1.append(sample_list)
    data_list1 = np.array(data_list1)
    # np.random.shuffle(data_list1)
    return data_list1


def output_inversed(scaler1, y_test1):
    arr = np.zeros((len(y_test1), 12))
    arr[:, 10:] = y_test1[:, :]
    arr_y_test = scaler1.inverse_transform(arr)
    arr_y_test = arr_y_test[:, 10:]
    # print(arr_y_test.shape)
    return arr_y_test


''' Load data '''
df_apple, df_google = data_loader()

print(df_apple)
print(df_google)

''' If filling values are necessary '''
# df_apple['Date'] = pd.to_datetime(df_apple['Date'])
# df_apple = df_apple.set_index('Date').asfreq('24h', method='bfill')
# df_apple['Date_col'] = df_apple.index
#
# df_google['Date'] = pd.to_datetime(df_google['Date'])
# df_google = df_google.set_index('Date').asfreq('24h', method='bfill')
# df_google['Date_col'] = df_google.index

''' Merge values '''
df = pd.merge(df_apple, df_google, how='inner', left_index=True, right_index=True)
print(df)

''' data and target '''
data = df[['High_x', 'Low_x', 'Open_x', 'Volume_x', 'Adj Close_x',
           'High_y', 'Low_y', 'Open_y', 'Volume_y', 'Adj Close_y', 'Close_x', 'Close_y']].to_numpy()
print(data.shape)

'''Scale data '''
from sklearn import preprocessing

# MinMax
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
print(data)

# Normalize
# data = preprocessing.normalize(data)
# print(data.shape)
# print(data.std(axis=0))

''' Generate time series data '''
period = 31
data_list = data_for_training(data, period)
print(data_list.shape)
target_list = np.zeros((len(data_list), 2))
for i in range(len(data_list)):
    value = data_list[i][30][10]
    value1 = data_list[i][30][11]
    target_list[i][0] = value
    target_list[i][1] = value1

data_list = data_list[:, :30, :]
print(data_list.shape)
print(target_list.shape)
print(target_list)

''' Split data '''
my_test_size = 0.1
ratio_train = int((1 - my_test_size) * len(data_list))
x_train = data_list[:ratio_train]
x_test = data_list[ratio_train:]
y_train = target_list[:ratio_train]
y_test = target_list[ratio_train:]
print(x_test.shape)

''' Build model '''
model = Sequential()
model.add(SimpleRNN(units=100, input_shape=x_train.shape[1:], activation="relu", recurrent_dropout=0.0))
# model.add(GRU(50, batch_input_shape=(2234, 30, 12)))
model.add(Dense(2, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

''' Training '''
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1, batch_size=30)

''' Results '''
results_train = model.evaluate(x_train, y_train)
results_test = model.evaluate(x_test, y_test)

print('train loss: {}, train acc: {}'.format(results_train[0], results_train[1]))
print('test loss: {}, test acc: {}'.format(results_test[0], results_test[1]))

plt.plot(history.history['loss'])
plt.title('(RNN) Loss plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

results_test = model.predict(x_test)
arr_y_test = output_inversed(scaler, y_test)
arr_results_test = output_inversed(scaler, results_test)
plt.plot(np.arange(len(results_test)), arr_y_test[:, 0], label='True label Apple')
plt.plot(np.arange(len(results_test)), arr_y_test[:, 1], label='True label Google')
plt.plot(np.arange(len(results_test)), arr_results_test[:, 0], label='predicted Apple')
plt.plot(np.arange(len(results_test)), arr_results_test[:, 1], label='predicted Google')
plt.legend()

plt.xlabel('Time in test (Day)')
plt.ylabel('Output value ($)')
plt.title('(RNN) Predicted vs. True label')
plt.show()

results = model.evaluate(x_test, y_test)
plt.plot(arr_y_test, arr_results_test, '*')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('(RNN) Predicted vs. True label')
plt.xticks(rotation=45, ha='right')
plt.show()

plt.plot(history.history['val_loss'])
plt.title('(RNN) (no dropout) Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

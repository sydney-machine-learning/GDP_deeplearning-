import copy
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import datetime
import time
import joblib
from datetime import timedelta, date
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

import seaborn as sns; sns.set_theme()

import errno

from numpy import array

# # Dataset prepocessing

data=pd.read_csv("/Users/rich/PycharmProjects/GDP_forecast/YEAR.csv")
data=data.drop(['Series Name', 'Series Code', 'Country Code'],axis=1)
data=data.drop([5,6,7,8,9],axis=0)
data.set_index('Country Name', inplace=True)
data=data.T
year=pd.period_range('1971','2019',freq='Y')
data.index= year
data = data['United States']

#
# total = data.isnull().sum().sort_values(ascending=False)
# print(total)
# print(data)

def rmse(pred, actual):
	return np.sqrt(((pred - actual) ** 2).mean())

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# define the model

def MODEL_LSTM(univariate, name, x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden, n_features):
    #if univariate is True:
	#	n_features = 1
	#else:
	#	n_features = 2 # can change

	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
	print(x_train.shape)
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
	print(x_test.shape)

	train_acc = np.zeros(Num_Exp)
	test_acc = np.zeros(Num_Exp)
	Step_RMSE = np.zeros([Num_Exp, n_steps_out])

	model = Sequential()
	model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features), dropout=0.2))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	model.summary()
	future_prediction = np.zeros([Num_Exp, 60])


	y_predicttest_allruns = np.zeros([Num_Exp, x_test.shape[0], x_test.shape[1]])

	print(y_predicttest_allruns.shape, ' shape ')


	Best_RMSE = 1000  # Assigning a large number

	start_time = time.time()
	for run in range(Num_Exp):
		print("Experiment", run + 1, "in progress")
		# fit model
		model.fit(x_train, y_train, epochs=Epochs, batch_size=10, verbose=0, shuffle=False)

		y_predicttrain = model.predict(x_train)
		y_predicttest = model.predict(x_test)
		y_predicttest_allruns[run,:,:] = y_predicttest
		#print(y_predicttest)
		train_acc[run] = rmse(y_predicttrain, y_train)
		#print(train_acc[run])

		test_acc[run] = rmse(y_predicttest, y_test)
		if test_acc[run] < Best_RMSE:
			Best_RMSE = test_acc[run]
			Best_Predict_Test = y_predicttest
		for j in range(n_steps_out):
			Step_RMSE[run][j] = rmse(y_predicttest[:, j], y_test[:, j])
	print("Total time for", Num_Exp, "experiments", time.time() - start_time)
	return future_prediction, train_acc, test_acc, Step_RMSE, Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns


# cov_mat = np.cov(data.T)
# print(cov_mat)
# ax = sns.heatmap(cov_mat)
# figure = ax.get_figure()
# figure.savefig('results.png', dpi=400)



n_steps_in = 3
n_steps_out = 3




train = data['1971':'2010']
test = data['2010':'2019']


x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)

x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
n_features = 1


Hidden = 10
Epochs = 50
n_steps_out = 3
n_steps_in = 3
name = 'Grid1'
Num_Exp = 30
univariate=True

future_prediction, train_acc, test_acc, Step_RMSE, Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns = MODEL_LSTM(
		univariate, name, x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden,
		n_features)

print(train_acc, test_acc)

mean_train = np.mean(train_acc, axis=0)
mean_test = np.mean(test_acc, axis=0)
std_train = np.std(train_acc, axis=0)
std_test = np.std(test_acc, axis=0)

step_rmse_mean = np.mean(Step_RMSE, axis=0)
step_rmse_std = np.std(Step_RMSE, axis=0)

print(mean_train, 'mean_train')
print(mean_test, 'mean_test')

print(std_train, 'std_train')
print(std_test, 'std_test')

print(step_rmse_mean, ' step_rmse mean')
print(step_rmse_std, ' step_rmse std')

results_combined = np.array([mean_train, mean_test, std_train, std_test])
results_combined = np.hstack((results_combined, step_rmse_mean))
results_combined = np.hstack((results_combined, step_rmse_std))
name=['mean_train', 'mean_test', 'std_train', 'std_test','step_rmse_mean','step_rmse_mean','step_rmse_mean','step_rmse_std','step_rmse_std','step_rmse_std']
results_combined=pd.DataFrame(index=name,data=results_combined)

# print(results_combined, ' results_combined ')
#


y_predicttest_mean = np.mean(y_predicttest_allruns, axis=0)
y_predicttest_std = np.std(y_predicttest_allruns, axis=0)

y_predicttest_low = np.percentile(y_predicttest_allruns, 5, axis=0)

y_predicttest_high = np.percentile(y_predicttest_allruns, 95, axis=0)

y_predicttest_meanstd = np.concatenate((y_predicttest_mean, y_predicttest_std, y_predicttest_low, y_predicttest_high),
									   axis=1)
print(y_predicttest_mean, 'y_predicttest_mean')
print(y_predicttest_std, 'y_predicttest_std')
print(y_predicttest_low, 'y_predicttest_low')
print(y_predicttest_high, 'y_predicttest_high')
print(y_predicttest_meanstd, 'std_testy_predicttest_meanstd')

np.savetxt('results/y_predicttest_meanstd_usa_.csv', y_predicttest_meanstd, delimiter = ',', fmt='%f')

np.savetxt('results/results_summary_usa_.csv', results_combined, delimiter = ',', fmt='%f')  # this



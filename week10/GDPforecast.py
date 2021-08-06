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
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.api import qqplot

from sklearn.metrics import mean_squared_error

import seaborn as sns;

import warnings
warnings.filterwarnings("ignore")

sns.set_theme()

import errno

from numpy import array


def rmse(pred, actual):
    return np.sqrt(((pred - actual) ** 2).mean())


# def mape(pred, actual):
#     return np.mean(np.abs((pred - actual) / actual)) * 100
#
#
# def smape(y_true, y_pred):
#     return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


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

def reshape_sequence(sequence, n_steps_in):
    X = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in

        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)

    return np.array(X)












# define the model

def MODEL_LSTM(x,x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden):
	n_features = 1
	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
	print(x_train.shape)
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
	print(x_test.shape)
	x = reshape_sequence(x,n_steps_in)
	x = x.reshape((x.shape[0], x.shape[1], n_features))

	train_acc = np.zeros(Num_Exp)
	test_acc = np.zeros(Num_Exp)
	Step_RMSE = np.zeros([Num_Exp, n_steps_out])

	model = Sequential()
	model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features)))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	model.summary()
	Best_RMSE = 1000  # Assigning a large number

	start_time = time.time()
	for run in range(Num_Exp):
		print("Experiment", run + 1, "in progress")
		# fit model
		model.fit(x_train, y_train, epochs=Epochs, batch_size=64, verbose=0, shuffle=False)
		y_predicttrain = model.predict(x_train)
		y_predicttest = model.predict(x_test)
		y_predict = model.predict(x)
		train_acc[run] = rmse(y_predicttrain, y_train)
		test_acc[run] = rmse(y_predicttest, y_test)
		if test_acc[run] < Best_RMSE:
			Best_RMSE = test_acc[run]
			Best_Predict_Test = y_predicttest
			Best_Predict = y_predict
		for j in range(n_steps_out):
			Step_RMSE[run][j] = rmse(y_predicttest[:, j], y_test[:, j])
	y = Best_Predict[[-3, -2, -1], :]
	y = y.reshape((y.shape[0], y.shape[1], n_features))
	y1_predict=model.predict(y)
	Best_Predict=np.row_stack((Best_Predict,y1_predict))
	print(Best_Predict.shape)


	print("Total time for", Num_Exp, "experiments", time.time() - start_time)
	return train_acc, test_acc, Step_RMSE, Best_Predict_Test, Best_Predict

def MULTI_MODEL_LSTM(x,x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden):
	n_features = 2
	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
	print(x_train.shape)
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
	print(x_test.shape)
	x = reshape_sequence(x,n_steps_in)
	x = x.reshape((x.shape[0], x.shape[1], n_features))

	train_acc = np.zeros(Num_Exp)
	test_acc = np.zeros(Num_Exp)
	Step_RMSE = np.zeros([Num_Exp, n_steps_out])

	model = Sequential()
	model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features)))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	model.summary()
	Best_RMSE = 1000  # Assigning a large number

	start_time = time.time()
	for run in range(Num_Exp):
		print("Experiment", run + 1, "in progress")
		# fit model
		model.fit(x_train, y_train, epochs=Epochs, batch_size=64, verbose=2, shuffle=False)
		y_predicttrain = model.predict(x_train)
		y_predicttest = model.predict(x_test)
		y_predict = model.predict(x)
		train_acc[run] = rmse(y_predicttrain, y_train)
		test_acc[run] = rmse(y_predicttest, y_test)
		if test_acc[run] < Best_RMSE:
			Best_RMSE = test_acc[run]
			Best_Predict_Test = y_predicttest
			Best_Predict = y_predict
		for j in range(n_steps_out):
			Step_RMSE[run][j] = rmse(y_predicttest[:, j], y_test[:, j])
	# Print('hh',Best_Predict)
	# y = Best_Predict[[-3, -2, -1], :]
	# y = y.reshape((y.shape[0], y.shape[1], n_features))
	# y1_predict=model.predict(y)
	# Best_Predict=np.row_stack((Best_Predict,y1_predict))
	# print(Best_Predict.shape)


	print("Total time for", Num_Exp, "experiments", time.time() - start_time)
	return train_acc, test_acc, Step_RMSE, Best_Predict_Test, Best_Predict








def Plot_Mean(name, Overall_Analysis, n_steps_out):
	labels = ['TrainRMSE(Mean)', 'TestRMSE(Mean)']
	Brazil = [Overall_Analysis[0][0], Overall_Analysis[0][5]]
	Canada = [Overall_Analysis[1][0], Overall_Analysis[1][5]]
	China = [Overall_Analysis[2][0], Overall_Analysis[2][5]]
	France = [Overall_Analysis[3][0], Overall_Analysis[3][5]]
	Germany = [Overall_Analysis[4][0], Overall_Analysis[4][5]]
	India = [Overall_Analysis[5][0], Overall_Analysis[5][5]]
	Indonesia = [Overall_Analysis[6][0], Overall_Analysis[6][5]]
	Italy  = [Overall_Analysis[7][0], Overall_Analysis[7][5]]
	Japan = [Overall_Analysis[8][0], Overall_Analysis[8][5]]
	Mexico = [Overall_Analysis[9][0], Overall_Analysis[9][5]]
	Turkey = [Overall_Analysis[10][0], Overall_Analysis[10][5]]
	United_Kingdom = [Overall_Analysis[11][0], Overall_Analysis[11][5]]
	United_States = [Overall_Analysis[12][0], Overall_Analysis[12][5]]

	yer1 = np.array([Overall_Analysis[0][3] - Overall_Analysis[0][0], Overall_Analysis[0][8] - Overall_Analysis[0][5]])
	yer2 = np.array([Overall_Analysis[1][3] - Overall_Analysis[1][0], Overall_Analysis[1][8] - Overall_Analysis[1][5]])
	yer3 = np.array([Overall_Analysis[2][3] - Overall_Analysis[2][0], Overall_Analysis[2][8] - Overall_Analysis[2][5]])
	yer4 = np.array([Overall_Analysis[3][3] - Overall_Analysis[3][0], Overall_Analysis[3][8] - Overall_Analysis[3][5]])
	yer5 = np.array([Overall_Analysis[4][3] - Overall_Analysis[4][0], Overall_Analysis[4][8] - Overall_Analysis[4][5]])
	yer6 = np.array([Overall_Analysis[5][3] - Overall_Analysis[5][0], Overall_Analysis[5][8] - Overall_Analysis[5][5]])
	yer7 = np.array([Overall_Analysis[6][3] - Overall_Analysis[6][0], Overall_Analysis[6][8] - Overall_Analysis[6][5]])
	yer8 = np.array([Overall_Analysis[7][3] - Overall_Analysis[7][0], Overall_Analysis[7][8] - Overall_Analysis[7][5]])
	yer9 = np.array([Overall_Analysis[8][3] - Overall_Analysis[8][0], Overall_Analysis[8][8] - Overall_Analysis[8][5]])
	yer10 = np.array([Overall_Analysis[9][3] - Overall_Analysis[9][0], Overall_Analysis[9][8] - Overall_Analysis[9][5]])
	yer11 = np.array([Overall_Analysis[10][3] - Overall_Analysis[10][0], Overall_Analysis[10][8] - Overall_Analysis[10][5]])
	yer12 = np.array([Overall_Analysis[11][3] - Overall_Analysis[11][0], Overall_Analysis[11][8] - Overall_Analysis[11][5]])
	yer13 = np.array([Overall_Analysis[12][3] - Overall_Analysis[12][0], Overall_Analysis[12][8] - Overall_Analysis[12][5]])

	width = 0.05  # the width of the bars
	Plot(name, labels, width, Brazil, Canada, China, France, Germany, India, Indonesia, Italy, Japan, Mexico, Turkey, United_Kingdom, United_States, yer1, yer2, yer3, yer4, yer5, yer6, yer7, yer8, yer9, yer10, yer11, yer12, yer13, "", "", "Train&Test_RMSE_Mean_Comparison", 4)

def Plot_Step_RMSE_Mean(name,Overall_Analysis,n_steps_out):
	Brazil=Overall_Analysis[0,10:n_steps_out*5+10:5]
	Canada=Overall_Analysis[1,10:n_steps_out*5+10:5]
	China=Overall_Analysis[2,10:n_steps_out*5+10:5]
	France=Overall_Analysis[3,10:n_steps_out*5+10:5]
	Germany=Overall_Analysis[4,10:n_steps_out*5+10:5]
	India=Overall_Analysis[5,10:n_steps_out*5+10:5]
	Indonesia=Overall_Analysis[6,10:n_steps_out*5+10:5]
	Italy=Overall_Analysis[7,10:n_steps_out*5+10:5]
	Japan=Overall_Analysis[8,10:n_steps_out*5+10:5]
	Mexico=Overall_Analysis[9,10:n_steps_out*5+10:5]
	Turkey=Overall_Analysis[10,10:n_steps_out*5+10:5]
	United_Kingdom=Overall_Analysis[11,10:n_steps_out*5+10:5]
	United_States=Overall_Analysis[12,10:n_steps_out*5+10:5]
	yer1=np.subtract(Overall_Analysis[0,13:n_steps_out*5+10:5],Brazil)
	yer2=np.subtract(Overall_Analysis[1,13:n_steps_out*5+10:5],Canada)
	yer3=np.subtract(Overall_Analysis[2,13:n_steps_out*5+10:5],China)
	yer4=np.subtract(Overall_Analysis[3,13:n_steps_out*5+10:5],France)
	yer5=np.subtract(Overall_Analysis[4,13:n_steps_out*5+10:5],Germany)
	yer6=np.subtract(Overall_Analysis[5,13:n_steps_out*5+10:5],India)
	yer7=np.subtract(Overall_Analysis[6,13:n_steps_out*5+10:5],Indonesia)
	yer8=np.subtract(Overall_Analysis[7,13:n_steps_out*5+10:5],Italy)
	yer9=np.subtract(Overall_Analysis[8,13:n_steps_out*5+10:5],Japan)
	yer10=np.subtract(Overall_Analysis[9,13:n_steps_out*5+10:5],Mexico)
	yer11=np.subtract(Overall_Analysis[10,13:n_steps_out*5+10:5],Turkey)
	yer12=np.subtract(Overall_Analysis[11,13:n_steps_out*5+10:5],United_Kingdom)
	yer13=np.subtract(Overall_Analysis[12,13:n_steps_out*5+10:5],United_States)
	labels = []
	for j in range(n_steps_out):
		labels=np.concatenate((labels,[str(j+1)]))
	width = 0.05  # the width of the bars
	Plot(name,labels,width, Brazil, Canada, China, France, Germany, India, Indonesia, Italy, Japan, Mexico, Turkey, United_Kingdom, United_States, yer1, yer2, yer3, yer4, yer5, yer6, yer7, yer8, yer9, yer10, yer11, yer12, yer13,"Steps","RMSE(Mean)","Step_RMSE_Comparison",2)

def Plot(name, labels, width, Brazil, Canada, China, France, Germany, India, Indonesia, Italy, Japan, Mexico, Turkey, United_Kingdom, United_States, yer1, yer2, yer3, yer4, yer5, yer6, yer7, yer8, yer9, yer10, yer11, yer12, yer13,
         xlabel, ylabel, Gname, cap):
	r1 = np.arange(len(labels))
	r2 = [x + width for x in r1]
	r3 = [x + width for x in r2]
	r4 = [x + width for x in r3]
	r5 = [x + width for x in r4]
	r6 = [x + width for x in r5]
	r7 = [x + width for x in r6]
	r8 = [x + width for x in r7]
	r9 = [x + width for x in r8]
	r10 = [x + width for x in r9]
	r11 = [x + width for x in r10]
	r12 = [x + width for x in r11]
	r13 = [x + width for x in r12]

	fig, ax = plt.subplots()
	rects1 = ax.bar(r1, Brazil, width, edgecolor='black', yerr=yer1, capsize=cap, label='Brazil')
	rects2 = ax.bar(r2, Canada, width, edgecolor='black', yerr=yer2, capsize=cap, label='Canada')
	rects3 = ax.bar(r3, China, width, edgecolor='black', yerr=yer3, capsize=cap, label='China')
	rects4 = ax.bar(r4, France, width, edgecolor='black', yerr=yer4, capsize=cap, label='France')
	rects5 = ax.bar(r5, Germany, width, edgecolor='black', yerr=yer5, capsize=cap, label='Germany')

	rects6 = ax.bar(r6, India, width, edgecolor='black', yerr=yer6, capsize=cap, label='India')
	rects7 = ax.bar(r7, Indonesia, width, edgecolor='black', yerr=yer7, capsize=cap, label='Indonesia')
	rects8 = ax.bar(r8, Italy, width, edgecolor='black', yerr=yer8, capsize=cap, label='Italy')
	rects9 = ax.bar(r9, Japan, width, edgecolor='black', yerr=yer9, capsize=cap, label='Japan')

	rects10 = ax.bar(r10, Mexico, width, edgecolor='black', yerr=yer10, capsize=cap, label='Mexico')
	rects11 = ax.bar(r11, Turkey, width, edgecolor='black', yerr=yer11, capsize=cap, label='Turkey')
	rects12 = ax.bar(r12, United_Kingdom, width, edgecolor='black', yerr=yer12, capsize=cap, label='United_Kingdom')
	rects13 = ax.bar(r13, United_States, width, edgecolor='black', yerr=yer13, capsize=cap, label='United States')

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xticks([r + width for r in range(len(China))], labels)
	plt.setp(ax.get_xticklabels(), fontsize=8)
	ax.legend()
	fig.tight_layout()
	plt.savefig("/Users/rich/PycharmProjects/GDP_forecast/Results/" + name + Gname + ".png", dpi=300)
	plt.show()


def ARIMA_evaluate(X,i):
	order = [(1, 1, 0), (0, 0, 1), (0, 2, 2), (0, 1, 2), (1, 0, 2),(0, 1, 0), (4, 1, 0), (0, 1, 2), (0, 1, 2), (6, 1, 0),(1, 0, 1), (4, 2, 2), (0, 1, 2)]
	size = int(len(X) * 0.75)
	train,test = X[0:size],X[size:len(X)]
	# history = [x for x in train]
#         predictions = list()
#     # walk-forward validation
#         for t in range(len(test)):
	model = ARIMA(train, order=order[i])
	model_fit = model.fit()
	arima_pred = model_fit.predict(start =3, end = 46, typ="levels")
	return arima_pred


# # Dataset prepocessing

data = pd.read_csv("/Users/rich/PycharmProjects/GDP_forecast/E7_G7.csv")
data = data.drop(['Units', 'Scale', 'Estimates Start After', 'Unnamed: 51'], axis=1)
data = data.drop([13, 14], axis=0)
print(data)
data.set_index('Country', inplace=True)
data = data.T
year = pd.period_range('1980', '2026', freq='Y')
data.index = year
df = data['1980':'2020']
Estimates_data = data['2021':'2026']

data1 = pd.read_csv("/Users/rich/PycharmProjects/GDP_forecast/GDP_CPI.csv")
data1 = data1.drop(['Units', 'Scale', 'Estimates Start After', 'Unnamed: 51'], axis=1)
data1 = data1.drop([26, 27], axis=0)

data1.set_index('Country', inplace=True)
data1 = data1.T
year1 = pd.period_range('1980', '2026', freq='Y')
data1.index = year1
df1 = data1['1980':'2020']
Estimates_data1 = data1['2021':'2026']

def main():
	order = [(1, 0, 0), (0, 0, 1), (0, 2, 2), (0, 1, 2), (1, 0, 2),(0, 1, 0), (4, 1, 0), (0, 1, 2), (0, 1, 2), (6, 1, 0),(0, 0, 0), (0, 0, 1), (0, 1, 2)]
	n_steps_in, n_steps_out = 3,3
	Overall_Analysis=np.zeros([13,10+n_steps_out*5])
	for i in range(1,14):
		Country=i
		if Country==1:
			data = df['Brazil']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Brazil']
			data1.columns = ['GDP','CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Brazil"
		if Country == 2:
			data = df['Canada']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Canada']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Canada"
		if Country == 3:
			data = df['China']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['China']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "China"
		if Country == 4:
			data = df['France']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['France']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "France"
		if Country == 5:
			data = df['Germany']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Germany']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Germany"
		if Country==6:
			data = df['India']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['India']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "India"
		if Country == 7:
			data = df['Indonesia']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Indonesia']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Indonesia"
		if Country == 8:
			data = df['Italy']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Italy']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Italy"
		if Country == 9:
			data = df['Japan']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Japan']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Japan"
		if Country==10:
			data = df['Mexico']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Mexico']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Mexico"
		if Country == 11:
			data = df['Turkey']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['Turkey']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "Turkey"
		if Country == 12:
			data = df['United Kingdom']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['United Kingdom']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "United Kingdom"
		if Country == 13:
			data = df['United States']
			TrainData = data['1980':'2010']
			TestData = data['2010':'2020']
			data1 = df1['United States']
			data1.columns = ['GDP', 'CPI']
			TrainData1 = data1['1980':'2010']
			TestData1 = data1['2010':'2020']

			name = "United States"

		x_train, y_train = split_sequence(TrainData, n_steps_in, n_steps_out)
		x_test, y_test = split_sequence(TestData, n_steps_in, n_steps_out)
		x_train1, y_train1 = split_sequence(TrainData1, n_steps_in, n_steps_out)
		x_test1, y_test1 = split_sequence(TestData1, n_steps_in, n_steps_out)
		y_train1 = y_train1.astype(np.float16)

		y_train1 = y_train1[:, :, 0]
		y_test1 = y_test1[:, :, 0]

		print(name)
		Num_Exp=30
		Epochs= 250
		Hidden=10
		train_acc = np.zeros(Num_Exp)
		test_acc = np.zeros(Num_Exp)
		Step_RMSE = np.zeros([Num_Exp, n_steps_out])
		train_acc, test_acc, Step_RMSE, Best_Predict_Test, Best_Predict = MODEL_LSTM(data,x_train, x_test, y_train, y_test, Num_Exp,
																		   n_steps_in, n_steps_out, Epochs, Hidden)
		train_acc1, test_acc1, Step_RMSE1, Best_Predict_Test1, Best_Predict1 = MULTI_MODEL_LSTM(data1,x_train1, x_test1, y_train1, y_test1, Num_Exp,
																		   n_steps_in, n_steps_out, Epochs, Hidden)


		arr = np.dstack((train_acc, test_acc))
		arr = arr.reshape(Num_Exp, 2)
		arr = np.concatenate((arr, Step_RMSE), axis=1)
		arr = arr.reshape(Num_Exp, 2 + n_steps_out)

		ExpIndex = np.array([])
		for j in range(Num_Exp):
			ExpIndex = np.concatenate((ExpIndex, ["Exp" + str(j + 1)]))

		ExpIndex1 = ['TrainRMSE', 'TestRMSE']
		for j in range(n_steps_out):
			ExpIndex1 = np.concatenate((ExpIndex1, ["Step" + str(j + 1)]))

		arr = np.round_(arr, decimals=5)
		arr = pd.DataFrame(arr, index=ExpIndex, columns=ExpIndex1)
		arr.to_csv("/Users/rich/PycharmProjects/GDP_forecast/Results/" + name  + "ExpAnalysis.csv")
		print(arr)

		Train_Mean = np.mean(train_acc)
		Train_Std = np.std(train_acc)
		Train_CI_LB = Train_Mean - 1.96 * (Train_Std / np.sqrt(Num_Exp))
		Train_CI_UB = Train_Mean + 1.96 * (Train_Std / np.sqrt(Num_Exp))

		Test_Mean = np.mean(test_acc)
		Test_Std = np.std(test_acc)
		Test_CI_LB = Test_Mean - 1.96 * (Test_Std / np.sqrt(Num_Exp))
		Test_CI_UB = Test_Mean + 1.96 * (Test_Std / np.sqrt(Num_Exp))

		Overall_Analysis[(i - 1)][0] = Train_Mean
		Overall_Analysis[(i - 1)][1] = Train_Std
		Overall_Analysis[(i - 1) ][2] = Train_CI_LB
		Overall_Analysis[(i - 1)][3] = Train_CI_UB
		Overall_Analysis[(i - 1) ][4] = np.min(train_acc)
		Overall_Analysis[(i - 1)][5] = Test_Mean
		Overall_Analysis[(i - 1) ][6] = Test_Std
		Overall_Analysis[(i - 1) ][7] = Test_CI_LB
		Overall_Analysis[(i - 1) ][8] = Test_CI_UB
		Overall_Analysis[(i - 1)][9] = np.min(test_acc)

		arr1 = np.vstack(([Train_Mean, Train_Std, Train_CI_LB, Train_CI_UB, np.min(train_acc), np.max(train_acc)],
						  [Test_Mean, Test_Std, Test_CI_LB, Test_CI_UB, np.min(test_acc), np.max(test_acc)]))

		for j in range(n_steps_out):
			Step_mean = np.mean(Step_RMSE[:, j])
			Step_std = np.std(Step_RMSE[:, j])
			Step_min = np.min(Step_RMSE[:, j])
			Step_CI_LB = Step_mean - 1.96 * (Step_std / np.sqrt(Num_Exp))
			Step_CI_UB = Step_mean + 1.96 * (Step_std / np.sqrt(Num_Exp))
			arr1 = np.vstack((arr1, [Step_mean, Step_std, Step_CI_LB, Step_CI_UB, Step_min, np.max(Step_RMSE[:, j])]))
			Overall_Analysis[(i - 1) ][5 * j + 10] = Step_mean
			Overall_Analysis[(i - 1) ][5 * j + 11] = Step_std
			Overall_Analysis[(i - 1) ][5 * j + 12] = Step_CI_LB
			Overall_Analysis[(i - 1) ][5 * j + 13] = Step_CI_UB
			Overall_Analysis[(i - 1) ][5 * j + 14] = Step_min
		arr1 = np.round_(arr1, decimals=5)
		arr1 = pd.DataFrame(arr1, index=ExpIndex1,
							columns=['Mean', 'Standard Deviation', 'CI_LB', 'CI_UB', 'Min', 'Max'])
		print(arr1)
		arr1.to_csv("/Users/rich/PycharmProjects/GDP_forecast/Results/" + name + "OverallAnalysis.csv")

		x_data = np.linspace(0, y_test.shape[0], num=y_test.shape[0])
		for j in range(n_steps_out):
			plt.figure()
			plt.plot(x_data, y_test[:, j], label='actual')
			plt.plot(x_data, Best_Predict_Test[:, j], label='predicted')
			plt.ylabel('RMSE')
			plt.xlabel('Time (samples)')
			plt.title('Actual vs Predicted')
			plt.legend()
			plt.savefig("/Users/rich/PycharmProjects/GDP_forecast/Results/" + name + 'pred_Step' + str(j + 1) + '.png', dpi=300)
			plt.show()
			plt.close()

		arima=ARIMA_evaluate(data, i=i-1)
		d = list(Best_Predict[:, 0])
		e = list(Best_Predict[-2:, 2])
		a=list(Best_Predict1[:, 0])
		b=list(Best_Predict1[-2:,2])
		c=d+e
		f=a+b
		year_1 = pd.period_range('1983', '2024', freq='Y')
		year_2 = pd.period_range('1983', '2026', freq='Y')
		year_3 = pd.period_range('1983', '2023', freq='Y')
		year_22 = pd.period_range('1980', '2026', freq='Y')
		year_33 = pd.period_range('1980', '2023', freq='Y')

		plt.figure()
		data.plot()
		arima.plot(label='ARIMA predicted')
		plt.ylabel('GDP Growth(%)')
		plt.xlabel('Year')
		plt.title('Actual vs Predicted')
		plt.legend()
		plt.savefig("/Users/rich/PycharmProjects/GDP_forecast/Results/" + name + 'ARIMA comparison'  + '.png', dpi=300)
		plt.show()
		plt.close()

		plt.figure()
		data.plot()
		plt.xlim(['1980','2026'])
		plt.plot(year_2, c, label='LSTM predicted')




		# plt.plot(year_2, arima, label='ARIMA predicted')
		plt.ylabel('GDP Growth(%)')
		plt.xlabel('Year')
		plt.title('Actual vs Predicted')
		plt.legend()
		plt.savefig("/Users/rich/PycharmProjects/GDP_forecast/Results/" + name + 'LSTM comparison'  + '.png', dpi=300)
		plt.show()
		plt.close()
		plt.figure()
		data.plot()
		plt.xlim(['1980','2023'])
		plt.plot(year_3, f, label='MULTI_LSTM predicted')



		# plt.plot(year_2, arima, label='ARIMA predicted')
		plt.ylabel('GDP Growth(%)')
		plt.xlabel('Year')
		plt.title('Actual vs Predicted')
		plt.legend()
		plt.savefig("/Users/rich/PycharmProjects/GDP_forecast/Results/" + name + 'MULTI comparison' + '.png', dpi=300)
		plt.show()
		plt.close()



		# Plot mean of train_RMSE and test_RMSE
		# Plot Std of train_RMSE and test_RMSE
	Plot_Mean(name, Overall_Analysis, n_steps_out)
		# Plot Step wise RMSE mean for different methods
	Plot_Step_RMSE_Mean(name, Overall_Analysis, n_steps_out)


	Overall_Analysis = Overall_Analysis.astype('float64')
	Overall_Analysis = np.round_(Overall_Analysis, decimals=4)
	Index1 = []
	for j in range(13):
		Index =  ['Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Indonesia',
       'Italy', 'Japan', 'Mexico', 'Turkey', 'United Kingdom',
       'United States']

	Column = [ 'Train-RMSE-Mean', 'Train-RMSE-Std', 'Train-CI-LB', 'Train-CI-UB', 'TrainRMSE-Min',
		  'Test-RMSE-Mean', 'Test-RMSE-Std', 'Test-CI-LB', 'Test-CI-UB', 'Test-RMSE-Min']
	for j in range(1, 4):
		Column = np.concatenate((Column,
							 ['Step' + str(j) + '-RMSE-Mean', 'Step' + str(j) + '-RMSE-Std', 'Step' + str(j) + '-CI-LB',
							  'Step' + str(j) + '-CI-UB', 'Step' + str(j) + '-RMSE-Min']))

	# Overall_Analysis = np.concatenate((Index, Overall_Analysis), axis=1)
	Overall_Analysis = pd.DataFrame(Overall_Analysis, index=Index, columns=Column)
	print(Column)
	print(Index)
	print(Overall_Analysis)
	Overall_Analysis.to_csv("/Users/rich/PycharmProjects/GDP_forecast/Results/OverallAnalysis.csv")


if __name__ == "__main__": main()
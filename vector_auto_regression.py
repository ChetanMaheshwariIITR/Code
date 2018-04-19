
# coding: utf-8




from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import pandas




#Loading dataset
dataframe = pandas.read_csv('Gold.csv',usecols=['price'], header=0)
dataframe=dataframe.dropna(axis=0, how='all')
dataframe=dataframe.fillna(method='bfill')
X = dataframe.values


#split dataset
train, test = X[1:len(X)-100], X[len(X)-100:]


# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]


#making Predictions
import math
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = math.sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.3f' % error)
from sklearn.metrics import r2_score
print(r2_score(test, predictions))


# plotting the results
pyplot.plot(test)
pyplot.plot(predictions, color='green',label='Silver')
pyplot.show()









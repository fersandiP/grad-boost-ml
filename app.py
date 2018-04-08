import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from sklearn import metrics
# from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score

def main():
	temp = lambda col: col not in ['payer_code']
	diabet = pd.read_csv('dataset_diabetes/diabetic_data.csv', na_values='?', usecols=temp)
	# print(diabet.isnull().sum())
	# print(diabet.dtypes)
	fill = {}
	for column in diabet:
		if diabet[column].dtypes == 'int64':
			fill[column] = diabet[column].mean()
		else:
			temp = diabet[column].mode()
			fill[column] = temp[0]
	diabet = diabet.fillna(value=fill)
	# print(diabet.isnull().sum())
	# print(np.sum(np.sum(automobile == '?')))
	
	# print(automobile[automobile != '?'].dropna())

	# automobile = automobile[automobile != '?'].dropna()
	# automobile = automobile.apply(pd.to_numeric, errors='ignore')
	# sns.set_style('whitegrid')
	# sns.barplot(x='body-style', y='price', hue='fuel-type', data=automobile) 
	# price_mean = automobile['price'].mean()
	# price_sd = np.std(automobile['price'])
	# plt.plot((-1,5),(price_mean,price_mean),'orange')
	# plt.plot((-1,5),(price_mean+price_sd, price_mean+price_sd),'red')
	# plt.plot((-1,5),(price_mean-price_sd, price_mean-price_sd),'yellow')
	# plt.show()

	# automobile = automobile[automobile['price'] <= price_mean+price_sd]
	# sns.barplot(x='body-style', y='price', hue='fuel-type', data=automobile) 
	# price_mean = np.mean(automobile['price'])
	# price_sd = automobile['price'].std()
	# plt.plot((-1,5),(price_mean,price_mean),'orange')
	# plt.plot((-1,5),(price_mean+price_sd, price_mean+price_sd),'red')
	# plt.plot((-1,5),(price_mean-price_sd, price_mean-price_sd),'yellow')
	# plt.show()

	# linreg = LinearRegression() #instantiate a new model
	# y = automobile['price']
	
	# automobile.corr(method='pearson')['price']
	# feature_cols = ['curb-weight', 'width', 'engine-size', 
	# 'length', 'horsepower', 'city-mpg', 'bore', 'height', 
	# 'compression-ratio', 'peak-rpm']
	
	# X = automobile[feature_cols]
	# linreg.fit(X, y) #fit the model to our data
	# print(pd.DataFrame({'Feature':feature_cols, 'Coefficient':linreg.coef_}))
	
	# y_pred = linreg.predict(X)
	# print(np.sqrt(metrics.mean_squared_error(y, y_pred)))

	# logreg = LogisticRegression()

	# automobile.corr(method='spearman')['price']
	# feature_cols = ['curb-weight', 'length', 'engine-size', 
	# 'horsepower', 'city-mpg', 'highway-mpg', 'width', 
	# 'wheel-base', 'bore', 'height']

	# X = automobile[feature_cols]
	# X_train, X_test, y_train, y_test = train_test_split(X, y)
	# logreg.fit(X_train, y_train)
	# print(logreg.score(X_test, y_test))

	# y_pred = logreg.score(X_test, y_test)
	# logreg.fit(X, y)
	# y_true = logreg.score(X, y)
	# print(y_true - y_pred)


if __name__ == '__main__':
	main()
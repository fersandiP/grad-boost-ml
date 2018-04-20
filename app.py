import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from sklearn import metrics
# from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.metrics import accuracy_score

def main():
	temp = lambda col: col not in ['payer_code']
	diabet = pd.read_csv('dataset/diabetic_data.csv', na_values='?', usecols=temp)
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
	temp = pd.DataFrame()
	for column in diabet:
		clean_df = diabet[column]
		elements = np.array(clean_df)
		if diabet[column].dtypes == 'int64':
			mean = np.mean(elements, axis=0)
			sd = np.std(elements, axis=0)
			final_list = [x for x in clean_df if (x > mean - 2 * sd)]
			final_list = [x for x in final_list if (x < mean + 2 * sd)]
			temp[column] = pd.Series(final_list)
		else:
			temp[column] = pd.Series(elements)
	# print(temp.head())
	# temp.sort_values(by=[''], ascending = False)

	# temp.to_csv('dataset/pre_processed.csv', encoding='UTF8', index=False)

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

	# train_gs_X, test_gs_X, train_gs_Y, test_gs_Y = train_test_split(new_features, target, random_state=42,train_size=0.1 )
	# gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
	#               'max_depth': [4, 6, 8],
	#               'min_samples_leaf': [20, 50,100,150],
	#               #'max_features': [1.0, 0.3, 0.1] 
	#               }
	# print(gb_grid_params)

	# gb_gs = GradientBoostingClassifier(n_estimators = 600)

	# clf = grid_search.GridSearchCV(gb_gs,
	#                                gb_grid_params,
	#                                cv=2,
	#                                scoring='roc_auc',
	#                                verbose = 3, 
	#                                n_jobs=10);
	# clf.fit(train_gs_X, train_gs_Y);

	# signal_event_no = counts = data[target == 1].count()[0]
	# background_event_no = counts = data[target == 0].count()[0]
	# ratio_background_to_signal = float(background_event_no)/signal_event_no
	# ratio_background_to_signal = numpy.round(ratio_background_to_signal, 3)
	# train_X, test_X, train_Y, test_Y = train_test_split(new_features, target, random_state=42,train_size=0.5 )              
	# gb6 = GradientBoostingClassifier( n_estimators=400, learning_rate=0.2,
	#    class_weight=ratio_background_to_signal, max_depth=6)

	# scores = cross_validation.cross_val_score(gb,
 #                                          all_data, target,
 #                                          scoring="roc_auc",
 #                                          n_jobs=6,
 #                                          cv=3);
	# "Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std())

	# params = {'n_estimators': 500, 'max_depth': 6,
	#         'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
	# clf = GradientBoostingRegressor(**params).fit(x_train, y_train)

	# mse = mean_squared_error(y_test, clf.predict(x_test))
	# r2 = r2_score(y_test, clf.predict(x_test))

	# print("MSE: %.4f" % mse)
	# print("R2: %.4f" % r2)


if __name__ == '__main__':
	main()

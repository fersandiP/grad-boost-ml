import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
temp = lambda col: col  in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diabetesMed']
diabet = pd.read_csv('dataset/pre_processed.csv',usecols=temp)
arrayDiabet = diabet.values
array = diabet.values
X = array[:,0:3]
Y = array[:,3]
# feature extraction
pca = PCA(n_components=2)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

print("------------------------------------------")
# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
X = array[:,0:4]
Y = array[:,4]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 2)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

print("-----------------------------------------")
# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
X = array[:,0:4]
Y = array[:,4]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

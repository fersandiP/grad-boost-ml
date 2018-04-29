import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
# load data
feature_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diabetesMed']
temp = lambda col: col in feature_cols
diabet = pd.read_csv('dataset/pre_processed.csv', usecols=temp)
array = diabet.values
length = len(feature_cols) - 1
X = array[:,0:length]
# feature extraction
pca = PCA(n_components=2)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
print("------------------------------------------")
# Feature Extraction with RFE
Y = array[:,length]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 2)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print("-----------------------------------------")
# Feature Importance with Extra Trees Classifier
model = ExtraTreesClassifier()
# feature extraction
model.fit(X, Y)
print(model.feature_importances_)
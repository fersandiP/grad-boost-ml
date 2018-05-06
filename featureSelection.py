import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


def feature_engineering(data):
    df = data.copy()
    def delete_column(del_col):
        for col in del_col: df.drop(col, 1, inplace=True)
            
    #Delete id column and label              
    delete_column(['encounter_id', "patient_nbr", "diabetesMed"])
    
    #Delete constant col
    df = df.loc[:, (df != df.iloc[0]).any()]
    
    #Delete categorical column more than 2 values (simplicity)
    excluded_col = ['gender', 'race', 'age']
    obj_col = [col for col in df.columns if df.dtypes[col] == 'O' and 
               len(df[col].unique()) > 2 and col not in excluded_col]
    delete_column(obj_col)
    
    #Convert age column to int range 0-9 (ordering)
    ages = df.age.unique()
    df['age'] = df['age'].map(dict((ages[i],i) for i in range(len(ages))))
    
    #Convert all Object column to int using one hot encoding
    df = pd.get_dummies(df)
    
    #Delete column with suffix No and id
    no_col = [col for col in df.columns if col.endswith("No") or col.endswith("id")]
    delete_column(no_col)
    
    return df


def main():
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


if __name__ == '__main__':
	main()

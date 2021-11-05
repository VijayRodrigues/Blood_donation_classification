import pandas as pd
import numpy as np
import pickle

blood_df = pd.read_csv('BloodDonation.csv')


X = blood_df.drop(['Made Donation in March 2007'], axis = 1)
y = blood_df['Made Donation in March 2007']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


from sklearn.tree import DecisionTreeClassifier
model_blood=DecisionTreeClassifier(max_leaf_nodes=4,max_features=3,max_depth=15)
model_blood.fit(X_train, y_train)

pickle.dump(model_blood , open('BloodPred.pkl', 'wb'))
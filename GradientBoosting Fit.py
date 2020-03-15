"""
Author: David Weck
Date Started: 3/11/2020

This part of the project consists of identifying best hyperparameters for 
a gradient boosted classifier and then fitting the best model to the 
NCAAB data 
"""
#Loading required packages
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

#Reading in the data and creating feature and target dfs
stats = pd.read_csv(r'.\stats_final.csv', index_col=0)
features = stats[['eFG%', 'TO%', 'OR%', 'FTR']]
target = stats['Result']

#Scaling all features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#Splitting data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, 
                                                    test_size=.2,
                                                    random_state=123
                                                    )
#Instantiating Gradient Boosting Classifier
GBC = GradientBoostingClassifier()

#Creating dictionary of hyperparameters
params = {'learning_rate' : [.01, .05, .1, .2],
          'n_estimators' : [100, 200, 300, 400, 500],
          'subsample' : [.5, .6, .7, .8, .9, 1],
          'min_samples_split' : [.02, .04, .06, .08, .1],
          'min_samples_leaf' : [.02, .04, .06, .08, .1],
          'max_depth' : [3,4,5,6],
          'max_features' : [2,3,4]}

#Using random search to find best hyperparameters
GBC_RS = RandomizedSearchCV(estimator=GBC,
                            param_distributions=params,
                            n_iter=15,
                            scoring='neg_log_loss',
                            cv=5,
                            random_state=123
                            )

#Fitting the RandomSearch object
GBC_RS.fit(X_train, y_train)

#Generating predictions and classification report
y_pred = GBC_RS.predict(X_test)
print(classification_report(y_test, y_pred))

#Identifying best hyperparameters
GBC_RS.best_params_
"""
Author: David Weck
Date Started: 2/11/2020

This part of the project consists of fitting logistic regression models
to the NCAAB data and selecting the best one
"""
#Loading required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

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

#Creating parameter grid to use in grid search
C = np.linspace(.1, 1, 10)
param_grid = {'C' : C}

#Instantiating Logistic Regression model
logreg = LogisticRegression(solver='lbfgs')

#Creating GridSearchCV object to select best hyperparameters
logreg_clf = GridSearchCV(estimator = logreg,
                           param_grid=param_grid,
                           scoring = 'neg_log_loss',
                           cv = 10,
                           )


#Fitting Grid Search object to the training data
logreg_clf.fit(X_train, y_train)

#Generating predictions and classification report
y_pred = logreg_clf.predict(X_test)
print(classification_report(y_test, y_pred))

#Identifying best hyperparameter
logreg_clf.best_params_
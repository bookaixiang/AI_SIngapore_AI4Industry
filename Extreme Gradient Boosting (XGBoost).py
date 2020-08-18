# Chapter 1: Classification with XGBoost
# Statquest intro to XGBoost: https://www.youtube.com/watch?v=OtD8wVaFm6E
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html

# XGBoost is  an ensemble method uses decision tress CART as its base learner. Recall ensemble use different models and
# aggregates their outputs to arrive at its final value

# Basic lib loads
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error

# Working Drive check and test
print(os.getcwd())
# os.chdir("")

# To tackle the fking Graphviz probs for tree graph have to specify the windows executable
# https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Datacamp example unable to download its datasets for churn_data

# Imaginary data from a ride-sharing app with user behaviors over their first month of app usage in a set of
# imaginary cities as well as whether they used the service 5 months after sign-up.

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:, :-1], churn_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: %f" % (accuracy))

# Base learners in XGB are decision treess, refresher portion

cancer = load_breast_cancer()

df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns=np.append(cancer['feature_names'], ['target']))
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
df.head()
df.info()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4 == y_test)) / y_test.shape[0]
print("accuracy:", accuracy)

# Cross validation with XGboost!
# In XGBoost the dataset has to be converted into Dmatrix. DMatrix is a internal data structure that used by XGBoost
# which is optimized for both memory efficiency and training speed.

# Test using other datasets
cancer = load_breast_cancer()

df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns=np.append(cancer['feature_names'], ['target']))

X, y = df.iloc[:, 0:-1], df.iloc[:, -1]

# Create the DMatrix from X and y: churn_DMatrix
cancer_dmatrix = xgb.DMatrix(data=X, label=y)

# Logistic regression is used as we are classifying if Cancer (True 1, False 0),
# the features are used to give scores that are weighted by importance (entropy?, more info gain to train
# the decision trees "ensemblebly" and combined the weights of each tree in xgb

params = {"objective": "reg:logistic", "max_depth": 3}

# Perform cross-validation: cv_results
# Perform 3-fold cross-validation by calling xgb.cv(). dtrain is your churn_dmatrix, params is your parameter dicti
# nfold is the number of cross-validation folds (3), num_boost_round is the number of trees we want to build (5),
# metrics is the metric you want to compute (this will be "error", which we will convert to an accuracy).

cv_results = xgb.cv(dtrain=cancer_dmatrix, params=params,
                    nfold=3, num_boost_round=5,
                    metrics="error", as_pandas=True, seed=123)

# cv_results stores the training and test mean and standard deviation of the error per boosting round (tree built)
# as a DataFrame. From cv_results, the final round 'test-error-mean' is extracted and converted into an accuracy,
# where accuracy is 1-error

print(cv_results)
# Compute average out-of-sample accuracy : Accuracy 1 - test error mean
print(((1 - cv_results["test-error-mean"]).iloc[-1]))

# Change metrics to AUC, show that abt 97% of the test instances were correctly classified
cv_results = xgb.cv(dtrain=cancer_dmatrix, params=params,
                    nfold=3, num_boost_round=5,
                    metrics="auc", as_pandas=True, seed=123)

print(cv_results)
print((cv_results["test-auc-mean"]).iloc[-1])  # The final round of boosting results AUC mean is quite good!

# CHAPTER 2: Regression with XGBoost
# FOr regression models we normally use the RMSE or MAE as the metrics for model perfomance
# In xgboost for supervised learning , the objective is to really find the model that yields the minimum
# value of the loss function which there are normally 3 scenarios based on the situations:
# A) reg:linear (old), reg:squarederror (new) - use for regression problems
# B) reg:logistic - use for classfication problems when you want just decision, not probability
# C) binary:logistic - use when you want probability rather than just decision

# Base learners in xgboost as we see previously can be Trees or linear
# blogpost for more details https://www.statworx.com/de/blog/xgboost-tree-vs-linear/
# https://xgboost.readthedocs.io/en/latest/parameter.html

# Loading Ames, Iowa housing dataset to predict sales prices (regression problem)
path = ("./Python DataCamp Slides/Extreme Gradient Boosting with XGBoost" +
        "/ames_housing_trimmed_processed.csv")
df = pd.read_csv(path)
df.info()

# Selecting out the target and features
y = df["SalePrice"]
X = df.drop(["SalePrice"], axis=1)

# Creating the training and test sets!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBRegressor: xg_reg objective is set for regression, our base learner (booster) are trees
# and using 10 trees
xg_reg = xgb.XGBRegressor(objective="reg:squarederror", booster="gbtree", n_estimators=10, seed=123)

# Fit training data and use model to predict on test set
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse, warning msg occurs as reg:linear is no longer used should be reg:squarederror
# rmse will be more sensitive to outliers, penalize due to the square component.
# https://www.youtube.com/watch?v=zMFdb__sUpw
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("Tree Based RMSE: %f" % (rmse))
# Means prediction on average will be about $28106 difference from the actual cost in that sort of sense

# VS Linear regerssion Base Learner
# Linear regression tree gbliner instead of tree

# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)

# Create the parameter dictionary: params
params = {"booster": "gblinear", "objective": "reg:squarederror"}

# Train the model: xg_reg
# um_boost_round and corresponds to the number of boosting rounds or trees to build
xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("LinearReg Base RMSE: %f" % (rmse))

# Tree based XGBoost seems to have learnt better, lower RMSE


# Evaluating model quality, base learner as linear regression!
housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective": "reg:squarederror", "max_depth": 4}

# Root mean squared error
# Perform cross-validation 4 FOld CV with 5 boosting rounds
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="rmse", as_pandas=True,
                    seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print(("Final round test-rmse-mean: %f" % (cv_results["test-rmse-mean"]).tail(1)))

# Mean absolute error
# Perform cross-validation but this time with Mean absolute error as the metric
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="mae", as_pandas=True,
                    seed=123)

print(cv_results)
# Extract and print final boosting round metric
print(("Final round test-mae-mean: %f" % (cv_results["test-mae-mean"]).tail(1)))

# Regularization and base learners in XGBoost!
# Recall that Regularization helps to control model complexity and how well it generalizes,
# Regularization parameters in XGBoost:
# gamma - minimum loss reduction allowed for a split to occur
# alpha - l1 regularization on leaf weights, larger values mean more regularization
# lambda - l2 regularization on leaf weight

# Create the DMatrix: housing_dmatrix
# Loading Ames, Iowa housing dataset to predict sales prices (regression problem)
path = ("./Python DataCamp Slides/Extreme Gradient Boosting with XGBoost" +
        "/ames_housing_trimmed_processed.csv")
df = pd.read_csv(path)
df.info()

# Selecting out the target and features
y = df["SalePrice"]
X = df.drop(["SalePrice"], axis=1)

# Creating the training and test sets!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

housing_dmatrix = xgb.DMatrix(data=X, label=y)
reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective": "reg:squarederror", "max_depth": 3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:
    # Update l2 strength
    params["lambda"] = reg

    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse",
                             as_pandas=True, seed=123)

    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))

# looks like as as the value of 'lambda' increases, so does the RMSE.


# Visualizing individual XGBoost trees and graphtrees
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective": "reg:squarederror", "max_depth": 2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree, aka the first index
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()

# Plot the last tree sideways using arguement rankdir
xgb.plot_tree(xg_reg, num_trees=-1, rankdir="LR")
plt.show()

# Can visually see the Decision tress at work, but have to understand more

# Visualizing feature importances: What features are most important in my dataset

housing_dmatrix = xgb.DMatrix(data=X, label=y)

params = {"objective": "reg:linear", "max_depth": 4}

xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the feature importances,
# This involves counting the number of times each feature is split on across all boosting rounds (trees) in the model,
# and then visualizing the result as a bar graph, with the features ordered according to how many times they appear.
xgb.plot_importance(xg_reg)
plt.show()

# GrLivArea seems to be the most important feature occuring across the trees


# CHAPTER 3: Fine-tuning your XGBoost model

# Create the DMatrix: housing_dmatrix
path = ("./Python DataCamp Slides/Extreme Gradient Boosting with XGBoost" +
        "/ames_housing_trimmed_processed.csv")
df = pd.read_csv(path)
df.info()

y = df["SalePrice"]
X = df.drop(["SalePrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

housing_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

# Create the parameter dictionary for each tree: params
params = {"objective": "reg:squarederror", "max_depth": 3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# With CV, Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse",
                        as_pandas=True, seed=123)

    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=["num_boosting_rounds", "rmse"]))
# increasing the number of boosting rounds seems to decreases the RMSE


# Automated boosting round selection using early_stopping

# Early stopping works by testing the XGBoost model after every boosting round against a hold-out dataset and
# stopping the creation of additional boosting rounds (thereby finishing training of the model early) if the
# hold-out metric ("rmse" in our case) does not improve for a given number of rounds. Here you will use the
# early_stopping_rounds parameter in xgb.cv() with a large possible number of boosting rounds (50). Bear in mind
# that if the holdout metric continuously improves up through when num_boost_rounds is reached, then early stopping
# does not occur.

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

# Create the parameter dictionary for each tree: params
params = {"objective": "reg:squarederror", "max_depth": 4}

# Perform cross-validation with early stopping: cv_results
# Use 10 early stopping rounds and 100 boosting rounds
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, early_stopping_rounds=10, num_boost_round=100,
                    metrics="rmse", as_pandas=True, seed=123)

# Boosting stops after round 49 where no improvement in rmse  occurs
print(cv_results)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
%matplotlib inline
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                      skiprows = 1, header = None) # skipping a row for test set

column_labels=['age','workingclass','fnlwgt','education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'wage_class']

train_set.columns=column_labels
test_set.columns=column_labels

#show the missing rows of ' ?' as the unknown values. going to drop them.
train_set.replace(' ?', np.nan).dropna().shape
test_set.replace(' ?', np.nan).dropna().shape

#perform the drop and save as new dataframes
train_nomiss=train_set.replace(' ?', np.nan).dropna()
test_nomiss=test_set.replace(' ?', np.nan).dropna()

#wage class column is dirty, clean with replace. getting rid of period. test to see if unique is clean
test_nomiss.wage_class=test_nomiss.wage_class.replace({' <=50K.': ' <=50K', ' >50K.':' >50K'})
test_nomiss.wage_class.unique()
train_nomiss.wage_class.unique()
#cleaning done, now XGBoosting

#Step 1: ordinal encoding to categoricals

combined_set=pd.concat([train_nomiss,test_nomiss], axis=0)

combined_set.info()

#looks good. now use Categorical codes from pandas.

for feat in combined_set.columns:
  if combined_set[feat].dtype =='object':
    combined_set[feat]=pd.Categorical(combined_set[feat]).codes #replaces str with ints

combined_set.info()
combined_set.head()
#looks good, converted strings to ints. since XGBoost only works on all integer datasets, we did the coding to allow it to work. Split the train and test sets into their new respective dataframes

final_train=combined_set[:train_nomiss.shape[0]] #nice trick to select up to last row
final_test=combined_set[train_nomiss.shape[0]:] #trick to select after last row of train

#setting up for XGB. model based on wage class.
y_train=final_train.pop('wage_class')
y_test=final_test.pop('wage_class')
y_train.head()
y_test.head()
y_train.unique()

#set up parameters for XGBoost, cv= cross validation, ind= index, GBM = gradient boosted model

cv_params1={'max_depth':[3,5,7], 'min_child_weight':[1,3,5]}

ind_params1={'learning_rate': 0.1 , 'n_estimators':1000, 'seed':123, 'subsample':0.8, 'colsample_bytree':0.8, 'objective':'binary:logistic' }


#create the model. GridSearchCV-Exhaustive search over specified parameter values for an estimator. Important members are fit, predict.
optimized_GBM= GridSearchCV(xgb.XGBClassifier(**ind_params1), cv_params1, scoring='accuracy', cv=5, n_jobs=-1)

#now run our grid search with 5 fold cross validation, see which params perform the best.

optimized_GBM.fit(final_train, y_train) 

#our output from the fitting of data: takes ~5 min to run.
#GridSearchCV(cv=5, error_score='raise',
#       estimator=xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
#       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
#       min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
#       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
#       scale_pos_weight=1, seed=123, silent=True, subsample=0.8),
#       fit_params={}, iid=True, n_jobs=-1,
#       param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7]},
#       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)


#by looking at the grid_scores_ we can determine with what accuracy of prediction we got
optimized_GBM.grid_scores_

#max is 86.778% with !!! max_depth of 3, min_child_weight of 5 !!!

#now play with subsampling params as well

cv_params2={'learning_rate': [0.1,0.01], 'subsample': [0.7,0.8,0.9]}
ind_params2={'n_estimatiors':1000, 'seed':123, 'colsample_bytree':0.8, 'objective':'binary:logistic', 'max_depth':3, 'min_child_weight':5}

optimized_GBM2 = GridSearchCV(xgb.XGBClassifier(**ind_params2), 
                            cv_params2, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM2.fit(final_train, y_train)

optimized_GBM2.grid_scores_

#we see with subsample params that best is 86.072% with params of learning_rate:0.1, subsample:0.7

#now with xgboost data matrix to provide early stopping CV
#params are max_depth of 3, min_child_weight of 5, learning_rate:0.1, subsample:0.7


xgdmat=xgb.DMatrix(final_train,y_train)#create DMatrix for XGB

xgb_params={'eta':0.1, 'seed':123, 'subsample':0.7, 'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':5}

#GridSearch optimized CV settings
#somehow y_train has values of 0 and 2. must fix.
cv_xgb=xgb.cv(params=xgb_params, dtrain=xgdmat, num_boost_round=3000, nfold=5, metrics=['error'], early_stopping_rounds=100)#look for early stopping that minimizes error

#look at the tail to see how accurate we got
print(cv_xgb.tail(5))
#got 88.3537% accurate!, took 542 rounds

#create final callable model

final_params={'eta':0.1, 'seed':123, 'subsample':0.7, 'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':5}

final_gb=xgb.train(final_params, xgdmat, num_boost_round=542)

#use seaborn to plot

sns.set(font_scale=1.5)

#plot feature importance. fnlwgt is most important with age and capital_gain coming second and third.
xgb.plot_importance(final_gb)

#make our own nice looking feature importance plot instead of using the builtin xgb.plot_importance

importances = final_gb.get_fscore()
importances

importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')

## model has now been tuned with cv grid search and early stopping. now test on test set!

testdmat=xgb.DMatrix(final_test)
y_pred=final_gb.predict(testdmat)
y_pred

#y_pred is outputted as probabilities by default, not class labels. To fix, we use:
y_pred=np.rint(y_pred)

accuracy_score(y_pred,y_test), 1-accuracy_score(y_pred,y_test)

#we are 86.746% accurate.

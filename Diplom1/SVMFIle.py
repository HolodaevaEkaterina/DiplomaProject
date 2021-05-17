import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.api as sm
import numpy as np
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn import svm
import pickle
from scipy import stats
import pandas as pd
from sklearn.linear_model import LogisticRegression



def svc(age, sex, days, criteri,rhytmc, rhytm):
  ekgTable = pd.read_csv("Ekg_table_full.csv")
  ekgTable = ekgTable.loc[ekgTable.Ritm.isna() == False]
  ekgTable = ekgTable.reset_index(drop = True)

  X = ekgTable.drop(["Code", "Group"], axis = "columns")
  y = ekgTable.Group
  print("Метод опорных векторов")



  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  clfSVM = svm.SVC()

  parametrs = {'kernel': ['poly', 'linear', 'rbf'], 'degree' : range(1,3), 'probability' : [True] }
  grid_search_cv_clf2 = GridSearchCV(clfSVM, parametrs, cv = 6)
  grid_search_cv_clf2.fit(X_train, y_train)
  print(grid_search_cv_clf2.best_params_)
  best_clf2 = grid_search_cv_clf2.best_estimator_
  print(best_clf2.score(X_test, y_test))

  dict = {"Age":pd.Series(age, index =[1]), "Sex":pd.Series(sex,index =[1]), "Day of Hospital":pd.Series(days,index =[1]),
            "Kriteriy":pd.Series(criteri, index =[1]), "RitmC":pd.Series(rhytmc, index =[1]), "Ritm":pd.Series(rhytm, index =[1])}
  d = pd.DataFrame(dict)
  print(X_test)
  print(d)
  result = best_clf2.predict_proba(d)
  return result[0][0]


def train():
  ekgTable = pd.read_csv("Ekg_table_full.csv")
  ekgTable = ekgTable.loc[ekgTable.Ritm.isna() == False]
  ekgTable = ekgTable.reset_index(drop=True)

  X = ekgTable.drop(["Code", "Group"], axis="columns")
  y = ekgTable.Group
  print("Метод опорных векторов")

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  clfSVM = svm.SVC()

  parametrs = {'kernel': ['poly', 'linear', 'rbf'], 'degree': range(1, 3), 'probability': [True]}
  grid_search_cv_clf2 = GridSearchCV(clfSVM, parametrs, cv=6)
  grid_search_cv_clf2.fit(X_train, y_train)
  print(grid_search_cv_clf2.best_params_)
  best_clf2 = grid_search_cv_clf2.best_estimator_
  print(best_clf2.score(X_test, y_test))
  save(best_clf2)

def save(best_clf2):
  with open('SVM', 'wb') as f:
    pickle.dump(best_clf2, f)

def Predict(age, sex, days, criteri,rhytmc, rhytm):
  with open('SVM', 'rb') as f:
    best_clf2 = pickle.load(f)
  dict = {"Age": pd.Series(age, index=[1]), "Sex": pd.Series(sex, index=[1]),
          "Day of Hospital": pd.Series(days, index=[1]),
          "Kriteriy": pd.Series(criteri, index=[1]), "RitmC": pd.Series(rhytmc, index=[1]),
          "Ritm": pd.Series(rhytm, index=[1])}
  d = pd.DataFrame(dict)
  print(X_test)
  print(d)
  result = best_clf2.predict_proba(d)
  return result[0][0]
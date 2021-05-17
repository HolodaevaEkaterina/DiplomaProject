import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.api as sm
import numpy as np
from sklearn.feature_selection import chi2
from sklearn import metrics, tree
from sklearn.tree import DecisionTreeClassifier
import math
from scipy import stats
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

pd.set_option('display.max_columns', None)

def tree1(age, sex, days, criteri,rhytmc, rhytm):
  ekgTable = pd.read_csv("Ekg_table_full.csv")
  ekgTable = ekgTable.loc[ekgTable.Ritm.isna() == False]
  ekgTable = ekgTable.reset_index(drop = True)

  X = ekgTable.drop(["Code", "Group"], axis = "columns")
  y = ekgTable.Group

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  clfTree = DecisionTreeClassifier()
  parametrs = {'max_depth':range(2,10), "criterion" : ['gini', 'entropy'], 'min_samples_leaf' : range(1, 3)}
  grid_search_cv_clf2 = GridSearchCV(clfTree, parametrs, cv = 6)
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

ekgTable = pd.read_csv("Ekg_table_full.csv", ';')
ekgTable = ekgTable.loc[ekgTable.Ritm.isna() == False]
ekgTable = ekgTable.reset_index(drop = True)

X = ekgTable.drop(["Code", "Group", "Kriteriy"], axis = "columns")
y = ekgTable.Group

table1 = pd.read_csv("Ekg_table_full.csv", ';')
table1 = table1.drop_duplicates("Code")
table2 = pd.read_excel("EKG.xlsx")
table2 = table2.drop_duplicates()

table2 = table2.rename(columns={'Код пациента':'Code','Группа':'Group'})


##table1=table1.drop(["Kriteriy"], axis = "columns")

df = pd.merge(table1, table2, on = ['Code','Group'])

df = df.loc[df.RR > 0]
df = df.loc[df.QT > 0]
df = df.reset_index()

df = df.drop(['index'], axis = 'columns')

df["NewKr1"] = 0

for index, row in df.iterrows():
  rr = df.RR.loc[df['Code'] == int(row["Code"])].sum()
  qt = df.QT.loc[df['Code'] == int(row["Code"])].sum()
  df.NewKr1.iloc[index] = math.log(rr, qt) * float(round(np.exp(df["Ritm"].iloc[index]), 5))



df = df.drop(["Code", "RR", "Ritm", "QT"], axis="columns")
df = df.drop_duplicates()
df.reset_index()


X = df.drop(["Group"], axis="columns")
y = df.Group
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clfTree = DecisionTreeClassifier()
parametrs = {'max_depth':range(2,7), "criterion" : ['gini', 'entropy'], 'min_samples_leaf' : range(1, 3)}
grid_search_cv_clf2 = GridSearchCV(clfTree, parametrs, cv = 6)
grid_search_cv_clf2.fit(X_train, y_train)
best_clf2 = grid_search_cv_clf2.best_estimator_
plt.figure(figsize=(40,20))  # customize according to the size of your tree
_ = tree.plot_tree(best_clf2)
plt.show()

#plt.figure(figsize=(40,20))
#_= df.groupby('NewKr1').agg({'NewKr1' : 'count'})
h = df['NewKr1'].hist()
fig = h.get_figure()
fig.show()


def train():
  ekgTable = pd.read_csv("Ekg_table_full.csv")
  ekgTable = ekgTable.loc[ekgTable.Ritm.isna() == False]
  ekgTable = ekgTable.reset_index(drop=True)

  X = ekgTable.drop(["Code", "Group"], axis="columns")
  y = ekgTable.Group

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  clfTree = DecisionTreeClassifier()
  parametrs = {'max_depth': range(2, 10), "criterion": ['gini', 'entropy'], 'min_samples_leaf': range(1, 3)}
  grid_search_cv_clf2 = GridSearchCV(clfTree, parametrs, cv=6)
  grid_search_cv_clf2.fit(X_train, y_train)
  print(grid_search_cv_clf2.best_params_)
  best_clf2 = grid_search_cv_clf2.best_estimator_
  print(best_clf2.score(X_test, y_test))
  save(best_clf2)

def save(best_clf2):
  with open('Tree', 'wb') as f:
    pickle.dump(best_clf2, f)

def Predict(age, sex, days, criteri,rhytmc, rhytm):
  with open('Tree', 'rb') as f:
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
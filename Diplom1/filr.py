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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import  classification_report, confusion_matrix
pd.set_option('display.max_columns', None)

table1 = pd.read_csv("Ekg_table_full.csv", ";")
table1 = table1.drop_duplicates("Code")
table2 = pd.read_excel("EKG.xlsx")
table2 = table2.drop_duplicates()

table2 = table2.rename(columns={'Код пациента':'Code','Группа':'Group'})

table3 = pd.read_excel("DnKG.xlsx")
table4 = pd.read_excel("DnOG.xlsx")
##table1=table1.drop(["Kriteriy"], axis = "columns")

df = pd.merge(table1, table2, on = ['Code','Group'])

df = df.loc[df.RR > 0]
df = df.loc[df.QT > 0]
df = df.reset_index()

df = df.drop(['index'], axis = 'columns')

df["NewKr1"] = 0

for index, row in df.iterrows():
  if row["Group"] == 0:
    count = len(table4[table4["Code"] == int(row["Code"])])
  else:
    count = len(table3[table3["Code"] == int(row["Code"])])
  rr = df.RR.loc[df['Code'] == int(row["Code"])].sum()
  qt = df.QT.loc[df['Code'] == int(row["Code"])].sum()
  df.NewKr1.iloc[index] = math.log((qt * rr * df["Ritm"].iloc[index]) / count, (rr / qt) * df["Ritm"].iloc[index])


df = df.drop([ "RR", "QT", "Day of hospital", "NewKr1"], axis="columns")

df = df.drop_duplicates()

df = df.drop(["Code"], axis="columns")
df.reset_index()

X = df.drop(["Group"], axis="columns")
y = df.Group
print(X)


listTree = []

for i in range(0, 100) :
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

  clfSVM = svm.SVC()
  parametrs = {'kernel': ['poly', 'linear', 'rbf'], 'degree': range(1, 3), 'probability': [True]}
  grid_search_cv_clf2 = GridSearchCV(clfSVM, parametrs, cv = 6)
  grid_search_cv_clf2.fit(X_train, y_train)
  print(grid_search_cv_clf2.best_params_)
  best_clf2 = grid_search_cv_clf2.best_estimator_
  listTree.append(best_clf2.score(X_test, y_test))
  print(best_clf2.score(X_test, y_test))

  if best_clf2.score(X_test, y_test) > 0.79 :
    y_pred_proba = best_clf2.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    y_pred = best_clf2.predict(X_test)
    '''importance = best_clf2.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()'''
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    #plt.show()
    #plt.show()
  #  feature_importances = best_clf2.feature_importances_
  #  feature_importances_df = pd.DataFrame({'features': list(X_train),
                             #               'feature_importances': feature_importances})
  #  feature_importances_df.sort_values('feature_importances', ascending=False)
   # print(feature_importances_df)


print(np.mean(listTree))
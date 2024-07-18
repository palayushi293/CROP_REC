import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path="C:\Users\palay\OneDrive\Desktop\python\Crop_recommendation (1) (1).zip"
df=pd.read_csv(path)
print(df.head())
print(df.tail())
print(df.describe())
print(df.info())

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

labe=LabelEncoder()
df['label']=labe.fit_transform(df['label'])
df['label']

df.isna().sum()

x=df.drop("label",axis=1)



y=df["label"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)
from sklearn.utils import resample

if x_train.shape[0] > y_train.shape[0]:
    y_train = resample(y_train, replace=True, n_samples=x_train.shape[0])
elif y_train.shape[0] > x_train.shape[0]:
    x_train = resample(x_train, replace=True, n_samples=y_train.shape[0])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


x_train.shape
x_train.shape
x_train.shape
y_train.shape

c_summary=pd.pivot_table(df,index=['label'],aggfunc='mean')
print(c_summary)

#SCALING

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
mms.fit(x_train)
x_train=mms.transform(x_train)
x_test=mms.transform(x_test)
x_test
x_train.shape
y_train.shape

#STANDARD
from sklearn.preprocessing import StandardScaler
sst=StandardScaler()
sst.fit(x_train)
x_train=sst.transform(x_train)
x_test=sst.transform(x_test)


#TRAIN
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier

from sklearn.metrics import accuracy_score

models = {
    'Logistic Regression': LogisticRegression(),

  'Naive Bayes':GaussianNB(),
  'Support Vector Machine':SVC(),
  'K-Nearest Neighbors':KNeighborsClassifier(),
  'Decision Tree':DecisionTreeClassifier(),
  'Random Forest':RandomForestClassifier(),
  'Bagging':BaggingClassifier(),
  'Adaboost':AdaBoostClassifier(),
  'Gradient Boosting':GradientBoostingClassifier(),
  'Extra Tress':ExtraTreeClassifier()



}
for name ,md in models.items():
  md.fit(x_train,y_train)
  ypred=md.predict(x_test)

  print(f"{name} with accuracy:{accuracy_score(y_test,ypred)}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\aisha\Downloads\logit classification.csv")

x=dataset.iloc[:, [2,3]].values
y=dataset.iloc[:, -1].values

# 80-20 TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=0)

#SCALE THE DATA
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# TRAIN THE MODEL X_TRAIN & Y_TRAIN

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

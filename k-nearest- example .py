import pandas as pd
from sklearn import preprocessing, model_selection, neighbors, svm
import numpy as np

tb=pd.read_csv('ecoli.data')
print(tb.head())
tb.drop(['name'],1,inplace=True)
#print(tb['name'])
tb.fillna(tb.mean(),inplace=True)

x = np.array(tb.drop(['class'],1))
y = np.array(tb['class'])
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

#clf=neighbors.KNeighborsClassifier(n_neighbors=9)
clf=svm.SVC()
clf.fit(x_train,y_train)

test_data=[[0.71,0.59,0.30,0.46,1.00,0.50,0.52]]
#test_data=test_data.reshape(len(test_data),-1)
print(clf.predict(test_data))
print(clf.score(x_test,y_test))

#print(x_train)
#print(x_test)

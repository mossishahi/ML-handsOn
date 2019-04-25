import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
import numpy as np
#print ('hello')


df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
print(df.head())
#df.drop(['id'])
df.drop(['id'],1,inplace=True)
print(df.head())

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
test=np.array([[1,2,3,4,5,5,6,6,0],[1,2,3,4,3,3,6,6,0],[3,3,6,4,5,8,4,4,1]])
#test=test.reshape(len(test),-1)
print(clf.predict(test))
accuracy=clf.score(X_test, y_test)
#print(accuracy)

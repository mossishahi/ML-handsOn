import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

#We take advantage of the data sets of Titanic passengers and attmept to cluster them in 2 Clusters. We have made use of this dataset 
# due to the clear clusters we can see before running any machine learning algorithms: Survived / UnSurvived

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print(train.describe())
print("\n")
print(test.describe())

#Firstly: impute NUMERICAL NA values
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

#Secondly: drop the categorical data we think have no effect on learning of k-means : 'Name','Ticket', 'Cabin','Embarked'
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

#Thirdly we convert the Sex value to a numerical value
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

#We should first drop the Survived values in train data
X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])


# Run the K-means algorithm 
kmeans = KMeans(n_clusters=2) 
kmeans.fit(X)

#We can change Optional parameters of K-means algorithm
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(X)

#Each time we test the algorithm accuracy using this bunch of code
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
		
		
# ================================================================================================
# Running the algorithm On Randomly generated data set
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
  
df = DataFrame(Data,columns=['x','y'])

# K-means with 3 Clusters
kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

# K-means with 3 Clusters
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)


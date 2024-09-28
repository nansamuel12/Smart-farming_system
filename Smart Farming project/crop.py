#import all necessery library

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# read dataset of crop and fertilizer

df=pd.read_csv('Crop_recommendation.csv')
#df2=pd.read_csv('Fertilizer_recommendation.csv')



#sample crop recommendation dataset
print(df.head())
print(df.tail())
print('                          ')



#catagorize the data by dependant variabel kind
print(df['label'].unique())
#print(df2['Fertilizer Name'].unique())

#heatmap for corelation diagram
sns.heatmap(df.corr(),annot=True)
#sns.heatmap(df2.corr(),annot=True)

#separating the dataset
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

#feature2=df2[['Temparature','Humidity','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous']]
#target2=df2[['Fertilizer Name']]

# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []


# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)




#Decesion Tree
from sklearn.tree import DecisionTreeClassifier
DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)
print(classification_report(Ytest,predicted_values))



from sklearn.model_selection import cross_val_score
# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)
score



import pickle
DT_pkl_filename = 'C:/python/DecisionTree.pkl'
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
DT_Model_pkl.close()

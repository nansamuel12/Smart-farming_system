
#import all necessery library for fertilizer recommendation 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")



# read dataset of crop and fertilizer

df2=pd.read_csv('Fertilizer_recommendation.csv')


#sample fertilizer recommendation dataset
# print(df2.head())
# print(df2.tail())


#catagorize the data by dependant variabel kind
print(df2['Fertilizer Name'].unique())

#heatmap for corelation diagram
#sns.heatmap(df.corr(),annot=True)
print(sns.heatmap(df2.corr(),annot=True))

#separating the dataset
continuous_data_cols = ["Temparature", "Humidity ", "Moisture", "Nitrogen", "Phosphorous"]
categorical_data_cols = ["Soil Type", "Crop Type"]

 
#Label Encoder the soil type data
soil_type_label_encoder = LabelEncoder()
df2["Soil Type"] = soil_type_label_encoder.fit_transform(df2["Soil Type"])
print(df2.head())

#Label Encoder for crop type
crop_type_label_encoder=LabelEncoder()
df2['Crop Type']=crop_type_label_encoder.fit_transform(df2['Crop Type'])
print(df2['Crop Type'])

#crop type dictionery 
croptype_dict = {}
for i in range(len(df2["Crop Type"].unique())):
    croptype_dict[i] = crop_type_label_encoder.inverse_transform([i])[0]
print(croptype_dict)


#soil type dictionery
soiltype_dict = {}
for i in range(len(df2["Soil Type"].unique())):
    soiltype_dict[i] = soil_type_label_encoder.inverse_transform([i])[0]
print(soiltype_dict)



#fertilizer name label encoder
#change the fertilizer catagorical data  into numerical data
fertname_label_encoder = LabelEncoder()
df2["Fertilizer Name"] = fertname_label_encoder.fit_transform(df2["Fertilizer Name"])




#fertilizer deictionery 
fertname_dict = {}
for i in range(len(df2["Fertilizer Name"].unique())):
    fertname_dict[i] = fertname_label_encoder.inverse_transform([i])[0]
print(fertname_dict)



X = df2[df2.columns[:-1]]
y = df2[df2.columns[-1]]

#count proportin of each fertilizer 
counter = Counter(y)



upsample = SMOTE()
X, y = upsample.fit_resample(X, y)
counter = Counter(y)
print(counter)

print(counter)








import pandas as pd
from sklearn.model_selection import train_test_split

#read from file
df=pd.read_csv('kidney_disease.csv')
#set columns of feature matrix
featureMatrix=df[['age', 'bp', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']]
#set columns for Y value
y=df['classification']
#Split data set into training and testing data
featureMatrix_train, featureMatrix_test, y_train, y_test =train_test_split(featureMatrix,y,test_size=0.30, random_state=42)

#When we train and test the same dataset, since it defeats the point of testing.
#When we train a dataset, we learn the specific trends of this data, so when we test the data it will predict based of the training, which gives a biased result.
#This causes the performance metrics to be less accurate, since the testing is based on previously trained data, instead of unknown data.

#The purpose of a testing set is to use unseen data, to ensure your model computes accurate and unbiased performance metrics.
#This ensures that your model accurately computes performance, rather than simply memorizing the data and providing biased computations.
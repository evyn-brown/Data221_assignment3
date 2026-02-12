import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


#read from file
df=pd.read_csv('kidney_disease.csv')

#set columns of feature matrix
featureMatrix=df[['age', 'bp', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']]

#set columns for Y value
y=df['classification']

#Split data set into training and testing data
featureMatrix_train, featureMatrix_test, y_train, y_test =train_test_split(featureMatrix,y,test_size=0.30, random_state=42)

#set K value
num_neighbors=5

#create KNN model for given K value
knn_model=KNeighborsClassifier(n_neighbors=num_neighbors)

#fit the model to our trained data
knn_model.fit(featureMatrix_train,y_train)

#make predictions on trained data
predicted_label=knn_model.predict(featureMatrix_test)

#compute and print confusion matrix
confusionMatrix=confusion_matrix(y_test, predicted_label )
print(f"confusionMatrix: {confusionMatrix}")

#compute and print accuracy score
accuracy=accuracy_score(y_test,predicted_label)
print(f"accuracy: {accuracy}")

#compute and print precision
precision=precision_score(y_test,predicted_label)
print(f"precision: {precision}")

#compute and print recall
recall=recall_score(y_test,predicted_label)
print(f"recall: {recall}")

#compute and print f1 score
f1score=f1_score(y_test,predicted_label)
print(f"f1score: {f1score}")

#True positive indicates that the person tested positive for kidney disease and actually has kidney disease.
#True negative means the person tested negative for kidney disease and does not have kidney disease.
#False Positive means the person tested positive for kidney disease but does not have kidney diease.
#False negative means the person tested negative for kidney disease but actually does have kidney disease.
#Accuracy does not fully evaluate a classification model since it may be misleading in imbalanced datasets, and fails to fully differentiate classification errors.
#The recall metric would be most important in this case, since it evaluates how many true positives there are within all positives. It measures the correctly identified individuals.
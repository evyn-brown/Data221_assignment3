import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#import file to dataframe
df=pd.read_csv('kidney_disease.csv')
#set columns of feature matrix
featureMatrix=df[['age', 'bp', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']]
#set columns for Y value
y=df['classification']
#Split data set into training and testing data
featureMatrix_train, featureMatrix_test, y_train, y_test =train_test_split(featureMatrix,y,test_size=0.30, random_state=42)

#set empty list to store accurace values
accuracy=[]
#list of k values to test
kValue=[1,3,5,7,9]

#iterate through each value in kValies
for i in kValue:
    #create KNN model for given K value
    knn_model = KNeighborsClassifier(n_neighbors=i)
    #fit the model to our trained data
    knn_model.fit(featureMatrix_train, y_train)
    #make predictions on trained data
    predicted_label = knn_model.predict(featureMatrix_test)
    #compute accuracy score
    accuracyScore=accuracy_score(y_test,predicted_label)
    #append accuracy score to accuracy list
    accuracy.append(accuracyScore)

#get results into a table
resultTable=pd.DataFrame({"K": kValue, "Accuracy": accuracy})
print(resultTable)

#When K=5, we get the highest accuracy.

#When we change the value of K, the accuracy tends to increase when we reduce variance (increase k), then will decrease again when variance is too high (large K).
#The accuracy follows a bell curve as increase K, the optimal accuracy is found somewhere in the middle.

#Small K values cause overfitting,since the KNN model becomes overly sensitive to variations in the trained data. This causes the model to overlook underlying patterns.

#Large K values cause underfitting, since the KNN model considers too many samples and begins to dismiss the underlying trends.


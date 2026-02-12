import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

#histogram
df=pd.read_csv('crime.csv')
plt.hist(df['ViolentCrimesPerPop'])
#set labels
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
plt.title("Frequency of Violent Crimes Per Population")
#plot grapg
plt.show()

#boxplot
df=pd.read_csv('crime.csv')
plt2.boxplot(df['ViolentCrimesPerPop'])
#set labels
plt2.xlabel("Violent Crimes Per Population")
plt2.ylabel("Frequency")
plt2.title("Frequency of Violent Crimes Per Population")
#plot graph
plt2.show()

#The histogram demonstrates a right skew, which indicates that there is overall a lower frequency of Violent Crimes per Population.
#The Histogram communicates that box with the most amount of datapoints is at x=~0.1 .

#The median is the middle value of the dataset, located centre/towards Q1.
#This indicates that there is a slight right skew of the data.

#The Box plot does not show evidence of outliers, as all values are within the 'whiskers', which extend from Q1-1.5(IQR) and Q3+1.5(IQR).
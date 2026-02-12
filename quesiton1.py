import pandas as pd
import numpy as np

#import file
df=pd.read_csv('crime.csv')

#find mean of ViolentCrimesPerPop
mean=np.mean(df['ViolentCrimesPerPop'])
#find median of ViolentCrimesPerPop
median=np.median(df['ViolentCrimesPerPop'])
#find STD of ViolentCrimesPerPop
std=np.std(df['ViolentCrimesPerPop'])
#find min of ViolentCrimesPerPop
min=np.min(df['ViolentCrimesPerPop'])
#find max of ViolentCrimesPerPop
max=np.max(df['ViolentCrimesPerPop'])
print(f"mean: {mean}")
print(f"median: {median}")
print(f"std: {std}")
print(f"min: {min}")
print(f"max: {max}")


#Since the mean and median are different values, the distribution of this data will be skewed to the right, since mean>median.
#The mean value is 0.44 while the minimum value is 0.02 and the maximum is 1.0, the standard deviation of 0.28 indicates that all the values are relatively close to each other.
#The min/max values would mainly affect the mean, since it takes the average of all values.
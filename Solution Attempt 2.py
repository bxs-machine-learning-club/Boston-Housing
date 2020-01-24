#At n=1000000, maximum accuracy of 0.9397590361445783
#And RMSE of 2.1027807177611155  

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np 
from scipy import stats 
from pprint import pprint

#Loads data
boston_dataset=load_boston()
data=pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
data['MEDV'] = boston_dataset.target

#Setting up predictor
correct=0 
zscore_tolerance=3 
pred_1=[]
data_split={
}
def round_zscore(d,mean_,stdev_,n):
  return int(round(n*(d-mean_)/(stdev_*zscore_tolerance)))

#Round data['MEDV']
for i in range(len(data)):
  data.at[i,'MEDV']=round(data.iloc[i]['MEDV'])

#Remove outliers
data=data[(np.abs(stats.zscore(data)) < zscore_tolerance).all(axis=1)]
data=data.reset_index()

#Drop CHAS
data=data.drop(axis=1,columns=['CHAS'])

#Scale each feature from a range from 1 to n 
#Note: It seems that increasing n increases accuracy, but only up to a point
for feature in data.columns[:-1]:
  Mean=np.mean(data[feature])
  Stdev=np.std(data[feature])
  for i in range(len(data)):
    data.at[i,feature]=round_zscore(data.iloc[i][feature],Mean,Stdev,1000000)

#Split up into feature-MEDV pairs
Min=min(data['MEDV'])
Max=max(data['MEDV'])
for feature in data.columns[:-1]:
  data_split[feature]=[{} for i in range(int(Max-Min+1))]
for i in range(len(data)): 
  for feature in data.columns[:-1]:
    A=str(round(float(data.iloc[i][feature])))
    B=int(data.iloc[i]['MEDV']-Min)
    C=data_split[feature][B]
    if A not in C:
      C[A]=0 
    data_split[feature][B][A]+=1

#Predictor:
#For each row data.iloc[i], go through each data.iloc[i][feature]
#Find the percentage of each MEDV in that feature that has the same data.iloc[i][feature]
#Find the MEDV with the highest percentage and put a point for that row's cell
#The MEDV with the highest points in the cell is the row's predicted MEDV
for i in range(len(data)):
  cell={}
  for feature in data.columns[:-1]:
    A=str(round(float(data.iloc[i][feature])))
    cell_1=[]
    #print(feature)
    for j in range(len(data_split[feature])):
      B=data_split[feature][j]
      if A not in B:
        B[A]=0
      if(sum(B.values())==0):
        B[A]+=1
      cell_1.append(B.get(A)/sum(B.values()))
    C=cell_1.index(max(cell_1))
    if str(C) not in cell:
      cell[str(C)]=0
    cell[str(C)]+=1
  pred=int(list(cell.keys())[list(cell.values()).index(max(cell.values()))])+Min
  pred_1.append(pred)
  if((pred)==data.iloc[i]['MEDV']):
    correct+=1
      
     
#Accuracy 
pprint(correct/len(data))

#Root Mean Squared Error
RMSE=0
for i in range(len(data)):
  RMSE+=(pred_1[i]-data.iloc[i]['MEDV'])**2
print((RMSE/len(data))**0.5)

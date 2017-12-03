'''
Created Sat Feb 13 2016 
@author vipulkhatana
'''
import numpy as np 
import pandas as pd

training_data = pd.read_csv("train_data_Softmax.csv",header=None)
training_labels = pd.read_csv("train_label_Softmax.csv",header=None)
test_data = pd.read_csv("test_data_Softmax.csv",header=None)

m = len(training_data)
k = 26
n = 16
lamda = 0.0001 # weight decay parameter
max_iterations = 100 
result = []


def ground_truth(i,j):
	if(training_labels.iloc[i,0]== j+1):
		return 1
	else:
		return 0	

def costFun(theta,matrix):	
	theta_grad = np.zeros((k,n))
	value = np.zeros(m)
	for j in range(0,k):
		print (j)
		for i in range(0,m):
			theta_x       = matrix[j][i]
			hypothesis    = np.exp(theta_x)
			sum = 0
			for  l in range(0,k):
				sum = np.exp(matrix[l][i])

			probabilities = hypothesis / sum
			value[i] = ground_truth(i,j) - probabilities;

		theta_grad[j] = -np.dot(np.transpose(training_data),value);
	
	theta_grad = theta_grad / m + lamda * theta
	
	return theta_grad

def Softmax():
	 
	print (m)
	print (k)

	theta = np.zeros((k,n))
	
	matrix = np.zeros((k,m))

	for i in range(1,30):
		matrix = np.dot(theta,training_data.T)
		theta = theta - lamda*costFun(theta,matrix)
  
      np.savetxt("trained.csv",result)


Softmax()


#*****TESTING*****###

test_data = pd.read_csv("test_data_Softmax.csv",header=None)
trained_theta = pd.read_cs("trained.csv",header=None)

result = []
def predict(theta,matrix):

	for i in range(0,m):
		maxVal = 0
		label = 0
		for j in range(0,k):
			theta_x = matrix[j][i]
			if theta_x > maxVal:
				maxVal = theta_x
				label = j
		result.append(label+1)
  
predict(trained_theta,np.dot(trained_theta,test_data.T))
np.savetxt("result_Softmax.csv",result)	 




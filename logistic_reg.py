'''
===================
Logistic Regression
===================

'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from mpl_toolkits.mplot3d import Axes3D

# Global variables
iteration = 0
Epsilon = 1000

''' To Read the X values '''
def Xread():
    with open ('train_data_logistic_reg.csv','rU') as f:
        reader = csv.reader(f)
        return np.matrix([map(float,line) for line in reader])

''' To Read the Y values '''
def Yread():
    with open ('train_label_logistic_reg.csv','rU') as f:
        reader = csv.reader(f)
        return np.matrix([map(float,line) for line in reader])
 
''' To create a theta vector based on dimension of input '''
def initialize_theta(x):
	return np.matrix([[float(0)]] * (x))

''' Obtains the Signum function of our hypothesis function '''
def signum(i, theta, x):
	return 1 / (1 + math.exp(-1*(x[i] * theta).item()))

''' Generate the Hessian '''
def Hessian(Theta, X, Y):
	D = np.diag([signum(i, Theta, X) * (1 - signum(i, Theta, X)) for i in range(X.shape[0])])
	return -(X.T * D * X) / X.shape[0]

''' Generate the Gradient of Log Likelihood '''
def grad_LL(Theta, X, Y):
	Hypo = np.matrix([[signum(i, Theta, X)] for i in range(X.shape[0])])
	return X.T * (Y - Hypo) / X.shape[0]

''' Calculates the Norm difference between two matrices '''
def norm(newTheta, Theta):
	print ('norn', np.linalg.norm(newTheta - Theta))
	return np.linalg.norm(newTheta - Theta)

''' Read input values '''
X = Xread()
Y = Yread()


X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X-X_mean)/X_std
X = np.c_[np.ones((X.shape[0], 1)), X]

''' Calculate Boundary using Newton's Method '''
Theta = initialize_theta(X.shape[1])

print (Theta)
while(True):
	iteration += 1
	newTheta = Theta - np.linalg.inv(Hessian(Theta, X, Y)) * grad_LL(Theta, X, Y)
	if norm(newTheta, Theta) < Epsilon and iteration > 1:
		break
	Theta = newTheta

''' Create 2D Plot of points and classification boundary '''
# Create two lists based on classification
X_one = []
X_zero = []
for i in range(len(Y.tolist())):
	if Y.item(i) > 1.5:
		X_one.append([X.tolist()[i][1], X.tolist()[i][2], X.tolist()[i][3]])
	else:
		X_zero.append([X.tolist()[i][1], X.tolist()[i][2], X.tolist()[i][3]])
  
def Imprt():
    with open ('test_data_logistic_reg.csv','rU') as f:
        reader = csv.reader(f)
        return np.matrix([map(float,line) for line in reader])
        
Z = Imprt()   
Z_mean = np.mean(Z, axis=0)
Z_std = np.std(Z, axis=0)
Z = (Z-Z_mean)/X_std
Z = np.c_[np.ones((Z.shape[0], 1)), Z]     
print (Z)

result=np.dot(Z,Theta);

for i in range(result.shape[0]):
    if (result.item(i)>1.5) :
            result.item(i) = 2 
    else:   result.item(i) = 1

np.savetxt("result_logistic_reg.csv", result, delimiter = ",");
    
    

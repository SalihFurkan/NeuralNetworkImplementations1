import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
'''
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import cross_validate
'''
# Sigmoid Function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def derivative_of_sigmoid(x):
	return np.multiply(sigmoid(x),(1-sigmoid(x)))

# Loading Matlab data
path = 'assign1_data1'
data = sio.loadmat(path)
# Storing data in a list
trainfull = data['trainims']
trainlbl  = data['trainlbls'][0]
testfull  = data['testims']
testlbl   = data['testlbls'][0]

train = []
test = []
for x in range(1900):
	train.append(trainfull[:,:,x].ravel()/255)
for x in range(1000):
	test.append(trainfull[:,:,x].ravel()/255)


train = np.vstack((np.matrix(train).T, -np.ones((1,1900))))
trainlbl = np.matrix(trainlbl)
test = np.vstack((np.matrix(test).T, -np.ones((1,1000))))
testlbl = np.matrix(testlbl)

#print(train[:,3])


# Initializing parameters
epoch_number = 100
mu, sigma, lr = 0, 0.01, 0.1 # mean, std, learning rate
w_firstlayer = np.matrix(np.random.normal(mu, sigma, size = (10,1025)))
w_secondlayer = np.matrix(np.random.normal(mu, sigma, size =(1, 11)))
meansqaurederror = np.zeros((1,epoch_number))

#print(w_secondlayer[:,:10], w_secondlayer)

# Training loop
for m in range(epoch_number):
	random_set = np.random.permutation(1900)
	epoch_error = 0
	error_test = 0

	for i in range(1,95):
		index = random_set[(i-1)*20:i*20]

		# Input Image
		Xp = train[:,index]
		# Desired Output
		dp = trainlbl[:,index]
		
		dw_second_batch = np.zeros((1,11))
		dw_first_batch  = np.zeros((10,1025))

		for j in range(1,20):
			X = Xp[:,j-1]
			target = dp[:,j-1]

			if target == 0:
				target = -1

			# Forward Propogation
			v_firstlayer = w_firstlayer*X
			Z = (np.tanh(v_firstlayer))
			neurons_forsecond = np.vstack((Z,-np.ones(1)))
			v_secondlayer = (w_secondlayer*neurons_forsecond)
			prediction = (np.tanh(v_secondlayer))
			
			# Back Propogation -- Gradient Descent
			local_gradient = (target - prediction) * (1-np.tanh(v_secondlayer)**2)
			dw_second = lr * local_gradient * neurons_forsecond.T

			w_hidden = w_secondlayer[:,:10]
			activation_derivative = np.diag((np.ones((10,1))-np.multiply(np.tanh(v_firstlayer),(np.tanh(v_firstlayer)))).A1)
			local_gradient_first = np.multiply((activation_derivative*w_hidden.T),local_gradient)
			dw_first = lr * local_gradient_first * X.T

			# Summing all the weight in a batch
			dw_first_batch = dw_first_batch + dw_first
			dw_second_batch = dw_second_batch + dw_second

			epoch_error = epoch_error + (np.square(target - prediction))

		dw_first_batch = dw_first_batch / 20
		dw_second_batch = dw_second_batch / 20

		# Updating Layers' Weight	
		w_secondlayer = w_secondlayer + dw_second_batch	
		w_firstlayer  = w_firstlayer + dw_first_batch

	meansqaurederror[:,m] = epoch_error/1900
'''
print(X.shape, test.shape)

# Plotting the error
t = np.arange(100)
plt.plot(t,meansqaurederror[0,:])
plt.show()

print(w_secondlayer)
test_result = []
for i in range(1000):
	testImage = np.matrix(np.zeros((1,1025)))
	testImage[:,:1024] = test[i,:]
	testImage[:,1024] = 1
	y_test = int(testlbl[:,i])

	v_test = (testImage*w.T)
	o_test = round(sigmoid(v))
'''
v_firstlayer = w_firstlayer*test[:,780]
Z = (np.tanh(v_firstlayer))
neurons_forsecond = np.vstack((Z,-np.ones(1)))
v_secondlayer = (w_secondlayer*neurons_forsecond)
prediction = (np.tanh(v_secondlayer))

print(prediction)

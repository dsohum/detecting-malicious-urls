import numpy as np
import math
import csv
# import ML
# import Dimentionality_Reduction
import os
import sys

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
#derivative of sigmoid wrt x
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:

    #constructor
    def __init__(self, layers, activation='sigmoid',feature_size=57):
        #||phi'(x)||=feature_size+1 we will include the bias as well
        if activation =='sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime

        
        #Weight Wij matrix i:l-1 layer-node output j:l layer-node 
        self.weights=[]
        #1st weight layer
        for i in range(len(layers)-2):
            r=2*np.random.random((layers[i]+1,layers[i+1]+1))-1
            # #print r.shape
            # setting weights for bias to 0
            r[0][:]=0
            self.weights.append(r)
            # #print r
        r=2*np.random.random((layers[len(layers)-2]+1,layers[len(layers)-1]))-1
        self.weights.append(r)
        
        
    #Training Neural Network
    def fit(self, X, y, learning_rate=0.05, epochs=10000,lambda_param=0.0001):
        #adding bias to the feature vectors
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T,X),axis=1)
        #print X.shape
        #training using forward-propogation and back-propogation
        #np.random.seed(0)
        for k in range(epochs):
            i=np.random.randint(X.shape[0])
            activation = [X[i]]
            
            #forward propogation
            for l in range(len(self.weights)):
                # for each layer
                ##print len(activation[l]), len(self.weights[l])
                dot_product = np.dot(activation[l],self.weights[l])
                # storing the activation for current layer (to be used in the -
                # - next iteration and back propogation
                activation.append(self.activation(dot_product))

            #  Important equations:
            #  delta(L)=(a(L)-y)*a'(L)
            #  delta(l)=(delta(l+1).weight(l+1))*a'(l)
            #  grad_ij(l)=a_i(l-1)delta_j(l)
            
            # computing delta(l) layer l for a data point, all units            
            error= activation[-1]-y[i]
            deltas = [error*self.activation_prime(activation[-1])]
            for l in range(len(self.weights)-2,-1,-1):
                delta = self.weights[l+1].dot(deltas[-1])*self.activation_prime(activation[l+1])
                deltas.append(delta)

            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(activation[i])
                #activation[0] is input
                delta = np.atleast_2d(deltas[i])
                self.weights[i] -= learning_rate*layer.T.dot(delta)
                #+lambda_param*self.weights[i]

            if (k+1)%1000 ==0:
                print 'epoch',k+1
                ##print "weights", self.weights

    def predict(self,x):
        x_1=np.concatenate((np.ones(1).T,x),axis=1)
        for l in range(0,len(self.weights)):
            x_1 = self.activation(x_1.dot(self.weights[l]))
            #print x_1
        return x_1
    

if __name__ == '__main__':
    print "Due to svm"
    os.system('python ML.py')
    print "Due to perceptron"
    os.system('python Dimentionality_Reduction.py')
    os.system('python Perceptron.py')
    N = NeuralNetwork([75,228,228,1])
    raw_data = open('svm_to_csv(0).csv')
    dataset = np.loadtxt(raw_data,delimiter=',')
    print "Due to Neural Network"
    X = dataset[:3000,1:76]
    y = dataset[:3000,:1]

    N.fit(X,y,epochs=6700)

    X_test=dataset[3000:11000,1:76]
    y_test=dataset[3000:11000,:1]
    # for i in range(len(X_test)):
        #print  1 if (N.predict(X_test[i])>0.5) else 0, y_test[i]
    prediction = np.array([ [0] if N.predict(x) <0.5 else [1] for x in X_test ])
    error = np.sum((prediction-y_test)**2)
    print "Due to neural net"
    print "error", error
    
    '''
    test_data=open('TestX.csv')
    dataset=np.loadtxt(test_data,delimiter=',')
    X=dataset[:,:54]
    prediction = [ 0 if N.predict(x)<0.5 else 1 for x in X]
    print prediction
    prediction = [ [str(i),str(prediction[i])] for i in range(len(prediction))]
            
    print "weights",N.weights
    i=1
    try:
        with open("TestY.csv",'wb') as CSVfile:
            CSVWriter = csv.writer(CSVfile)
            CSVWriter.writerow(['Id','Label'])
            for row in prediction:
                CSVWriter.writerow(row)
    except(IOError) as e:
        print("Error in opening file!{0}!",format(e.strerror))
    except:
        print(sys.exc_info()[0])
   '''

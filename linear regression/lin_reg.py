import sys
import numpy as np
import matplotlib.pyplot as plt
import time

def dataLoad(filename):
  # Declare an empty list
  Arr = []

  # Attempt to open file
  with open(filename, 'r') as f:
    # For each line in the file
    for line in f:
    #Get rid of unwanted lines
      line = line.strip()
    #Replace all recurring empty spaces
      line = line.replace("  ", " ")
      line = line.replace("  ", " ")
      data = line.split(" ") # Split string into elements using delimiter
      Arr.append(data) # Add data into arr

  X = np.array(Arr).astype(np.float)
  return X
    
def dataNorm(X):
    min = np.min(X[:,0:13],axis=0)
    max = np.max(X[:,0:13],axis=0) 
    X_norm = (X[:,0:13]-min)/(max-min) 

    X_norm = np.insert(X_norm,0,1, axis = 1)
    X_norm = np.insert(X_norm,[14],X[:, 13:14], axis = 1)
    return X_norm
    
def errCompute(X_norm, Theta):
    m = X_norm.shape[0]
    yHat = np.dot(X_norm[:, 0:14],Theta)
    sumOfY = np.dot( (X_norm[:, 14:15] - yHat).T, (X_norm[:, 14:15] - yHat))
    result =  sumOfY / (2 * m )
    return result

def sumOfSquareErrors(X, theta):
  errorVal = 0.0
  # Get vector of predicted values using weights
  predictedY = np.dot(X[:, 0:14], theta)
  # Calculate square of all errors
  for i in range(X.shape[0]):
    errorVal += np.float(np.square(X[i:i + 1, 14:15] - predictedY[i][0]))  
  return errorVal

def gradientDescent(X_norm, theta, alpha, num_iters):
    DeltaW = np.copy(theta)
    RMSEArr = np.zeros((num_iters, 2))
    Features =  theta.shape[0]
    Samples = X_norm.shape[0]
    for i in range(num_iters) :
        yHat = np.dot(X_norm[:, 0:14],theta)
        DeltaY = X_norm[:, 14:15] - yHat
        for j in range(Features):
            DeltaW[j] = np.dot(DeltaY.T, X_norm[:, j:j+1])
            DeltaW[j] /= Samples
        theta += (DeltaW * alpha)
        # RMSEArr[i][0] = i
        # RMSEArr[i][1] = rmse(yHat, X_norm[:, 14:15])
    # plt.plot(RMSEArr[:,0:1],  RMSEArr[:,1:2])
    # plt.title('RMSE')
    # plt.ylabel('Error')
    # plt.xlabel('Iterations')
    # plt.savefig('output/RMSE.png')
    # plt.show()
    return theta

def rmse(testY, stdY):
    N = testY.shape[0]
    sumOfY = 0
    for i in range (N):
        sumOfY += (testY[i] -  stdY[i])**2
    sumOfY /= N
    sumOfY = sumOfY**(0.5)
    return sumOfY

def kfoldCrossValidation(k, X_norm, theta, alpha, num_iters):
    Error = 0
    Splits = np.array_split(X_norm, k)
    for i in range (k):
        trainingFolds = dataLoad('data/Training'+ str(i) + '.data')
        testFold = dataLoad('data/Testing'+ str(i) + '.data')
        theta = gradientDescent(trainingFolds, np.zeros((14,1)), alpha, num_iters)
        Error += sumOfSquareErrors(testFold, theta )
    Error = np.sqrt(Error / X_norm.shape[0] )
    return Error


def WriteSplitData(k, X_norm):
    Splits = np.array_split(X_norm, k)
    for i in range (k):
        TestFold = np.empty((0,X_norm.shape[1]))
        for others in range (k):
            if(others == i):
                continue
            TestFold = np.append(TestFold,Splits[others],axis = 0)
        np.savetxt('data/Training' + str(i) + '.data', TestFold , delimiter=' ', fmt='%f')
        np.savetxt('data/Testing' + str(i) + '.data', Splits[i] , delimiter=' ', fmt='%f')

def FeaturePricePlot(X):
    #Saving for each feature
    #ZN
    plt.plot(X[:,0:1],  X[:,13:14], 'ob')
    plt.title('ZN-PricePlot')
    plt.ylabel('Price')
    plt.xlabel('Normalized Data')
    plt.savefig('output/ZN-PricePlot.png')
    plt.show()
    #INDUS
    plt.plot(X[:,1:2],  X[:,13:14], 'or')
    plt.title('INDUS-PricePlot')
    plt.ylabel('Price')
    plt.xlabel('Normalized Data')
    plt.savefig('output/INDUS-PricePlot.png')
    plt.show()
    #CHAS
    plt.plot(X[:,2:3],  X[:,13:14], 'og')
    plt.title('CHAS-PricePlot')
    plt.ylabel('Price')
    plt.xlabel('Normalized Data')
    plt.savefig('output/CHAS-PricePlot.png')
    plt.show()
    #NOX
    plt.plot(X[:,3:4],  X[:,13:14], 'oy')
    plt.title('NOX-PricePlot')
    plt.ylabel('Price')
    plt.xlabel('Normalized Data')
    plt.savefig('output/NOX-PricePlot.png')
    plt.show()
    #RM
    plt.plot(X[:,4:5],  X[:,13:14], 'om')
    plt.title('RM-PricePlot')
    plt.ylabel('Price')
    plt.xlabel('Normalized Data')
    plt.savefig('output/RM-PricePlot.png')
    plt.show()

    #Plot all into 1 graph
    plt.plot(X[:,0:1],  X[:,13:14], 'ob')
    plt.plot(X[:,1:2],  X[:,13:14], 'or')
    plt.plot(X[:,2:3],  X[:,13:14], 'og')
    plt.plot(X[:,3:4],  X[:,13:14], 'oy')
    plt.plot(X[:,4:5],  X[:,13:14], 'om')
    plt.title('ALL 5 FEATURES-PricePlot')
    plt.ylabel('Price')
    plt.xlabel('Normalized Data')
    plt.legend(['ZN', 'INDUS', 'CHAS','NOX', 'RM'])
    plt.savefig('output/AllFeature-PricePlot.png')
    plt.show()
def main():
    X = dataLoad('housing.data')
    #Comment functions used for plots
    # FeaturePricePlot(X)
    X_norm = dataNorm(X)
    Error = errCompute(X_norm,  np.zeros((X_norm.shape[1]-1,1)))
    theta = gradientDescent(X_norm, np.zeros((14,1)),0.01,1500)
    #WriteSplitData(5,X_norm)

    kfoldCrossValidation(5,X_norm, np.zeros((14,1)),0.01,1500)
    #commented out to prevent printing graph
    # ErrorAlpha = [0.01, 0.1, 0.5]
    # ErrorValue = []
    # for i in range (len(ErrorAlpha)):
    #     ErrorValue.append(kfoldCrossValidation(5,X_norm, np.zeros((14,1)),ErrorAlpha[i],1500))
    # plt.plot(ErrorAlpha, ErrorValue, linewidth=2.0)
    # plt.xlabel('Alpha')
    # plt.ylabel('Error')
    # plt.title('Error for each Alpha')
    # plt.savefig('output/CrossReferenceAlphaErrors.png')
    # plt.show()
    
main()











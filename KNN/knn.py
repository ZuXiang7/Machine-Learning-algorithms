import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#lists for classification report
allPred = []
allTrue = []
def loadData(filename):
  # load data from filename into X
  X=[]
  count = 0
  
  text_file = open(filename, "r")
  lines = text_file.readlines()
    
  for line in lines:
    X.append([])
    words = line.split(",")
    # convert value of first attribute into float  
    for word in words:
      if (word=='M'):
        word = 0.333
      if (word=='F'):
        word = 0.666
      if (word=='I'):
        word = 1
      X[count].append(float(word))
    count += 1
  return np.asarray(X)

def dataNorm(X):

    min = np.min(X[:,0:8],axis=0)#min of all features
    max = np.max(X[:,0:8],axis=0) #max of all features
    X_norm = (X[:,0:8]-min)/(max-min) #normalize features

    X_norm = np.insert(X_norm,[8],X[:, 8:9], axis = 1)#insert output to last row
    return X_norm

def testNorm(X_norm):
  xMerged = np.copy(X_norm)
  #merge datasets
  # for i in range(len(X_norm)-1):
  #   xMerged = np.concatenate((xMerged,X_norm[i+1]))
  print (np.mean(xMerged,axis=0))
  print (np.sum(xMerged,axis=0))

def splitTT(X_Norm, percentTrain):
  TrainRows = percentTrain * X_Norm.shape[0]#number or rows for training
  #shuffle the matrix
  ShuffledArray = np.copy(X_Norm)
  np.random.shuffle(ShuffledArray)
  #seperate into training and test matrix
  train = ShuffledArray[0: int(TrainRows)]
  test = ShuffledArray[int(TrainRows):  ]
  #return list of them
  return [train, test]

def splitCV(X_Norm, k):
  #shuffle the array
  ShuffledArray = np.copy(X_Norm)
  np.random.shuffle(ShuffledArray)
  #split them
  splits = np.array_split(ShuffledArray, k)
  return splits

def KNN(train,test,k):
  truePositive = 0 #counter for number of true positive
  col = test.shape[1] - 1 #to not include output
  #get number of rows for testing and training
  testRows = test.shape[0]
  trainRows = train.shape[0]
  #duplicate test and train data to do in one shot
  testRepeated = np.repeat(test[:, 0:8],train.shape[0], 0)#repeat because treated as outer loop
  trainTiled = np.tile(train[:, 0:8],(test.shape[0],1))#tile because treated inner loop
  distArr = testRepeated - trainTiled#get all point difference (X1 - X2) for each feature
  distArr = np.square(distArr)#square all points (X1 - X2)^2 for each feature
  sumedArr = np.sum(distArr,axis = 1)#get summation of each row  for Σ(X1 - X2)^2
  sqrtArr = np.sqrt(sumedArr)# √(Σ(X1 - X2)^2)
  sqrtArr = np.expand_dims(sqrtArr, axis=1)#for appending
  #for appending test and train outputs
  allTrainResult = np.tile(train[:, 8:9],(test.shape[0],1))
  allTestResult = np.repeat(test[:, 8:9],train.shape[0], 0)
  sqrtArr = np.append(sqrtArr,allTrainResult, axis = 1 )#append train result
  sqrtArr = np.append(sqrtArr,allTestResult, axis = 1  )#append test result
  #split for each test
  allArrs = np.array_split(sqrtArr,test.shape[0])
  #for all test
  for i in range(testRows): #input
    currArr = allArrs[i]
    #sort by 1st column, the distance
    currArr = np.sort(currArr.view('i8,i8,i8'),order=['f0'], axis = 0).view(np.float)
    ArrayOfK = currArr[0:k, :]#get closest k outputs
    if(k > 1):
      nearest_neighbours_k_outputs = np.squeeze(ArrayOfK[:, 1:2])#copy all output
      nearest_neighbours_k_outputs = nearest_neighbours_k_outputs.astype(int)
      counts = np.bincount(nearest_neighbours_k_outputs)#find highest count of output
      final_predicted_value = np.argmax(counts)#find output with highest count
    else:
      final_predicted_value = ArrayOfK[0][1]
    if(final_predicted_value == ArrayOfK[0][2]):#if predicted same as output
      truePositive += 1
    #for classification report
    allPred.append(final_predicted_value)
    allTrue.append(ArrayOfK[0][2])
  return truePositive / testRows


def CrossValidateKNN(SplitX, kfold, k):
  AllAcc = np.zeros((kfold,1))#to hold all accuracy
  for i in range (kfold):
    testArray = SplitX[i]#for testing
    trainArray = np.empty((0,SplitX[0].shape[1]))#to be filled with arrays not used to testing
    for j in range (kfold):#append all arrays to trainArray
      if(j == i):#dont append array currently used for testing
        continue
      trainArray = np.append(trainArray,SplitX[j],axis = 0)
    AllAcc[i][0] = KNN(trainArray,testArray,k)
  return np.mean(AllAcc)#return avg of all accuracy


# this is an example main for KNN with train-and-test + euclidean
def knnMain(filename,percentTrain,k):
  #data load
  X = loadData(filename)
  X_Norm = dataNorm(X)
  #for classification report
  k = 15
  kfold = 5
  CV5 = splitCV(X_Norm, kfold)
  accuracy = CrossValidateKNN(CV5,kfold,k)
  print(classification_report(allTrue,allPred))

  #Get time - Commnted out as it takes long
  # TimeTT7 = []
  # TimeTT6 = []
  # TimeTT5 = []

  # TimeCV5 = []
  # TimeCV10= []
  # TimeCV15 = []
  # K_Iteration = [1,5,10,15,20]
  # k = 1
  # TT7 = splitTT(X_Norm, 0.7)
  # TT6 = splitTT(X_Norm, 0.6)
  # TT5 = splitTT(X_Norm, 0.5)

  # CV5 = splitCV(X_Norm, 5)
  # CV10 = splitCV(X_Norm, 10)
  # CV15 = splitCV(X_Norm, 15)
  # for i in range(5):
  #   k = K_Iteration[i]
  #   print("k = ", k)
  #   start = time.time()
  #   KNN(TT7[0],TT7[1],k)
  #   end = time.time()
  #   TimeTT7.append(end - start)
  #   print("TimeTT7 = ",end - start )

  #   start = time.time()
  #   KNN(TT6[0],TT6[1],k)
  #   end = time.time()
  #   TimeTT6.append(end - start)
  #   print("TimeTT6 = ",end - start )

  #   start = time.time()
  #   KNN(TT5[0],TT5[1],k)
  #   end = time.time()
  #   TimeTT5.append(end - start)
  #   print("TimeTT5 = ",end - start )
  #   #cv
  #   start = time.time()
  #   CrossValidateKNN(CV5,5,k)
  #   end = time.time()
  #   TimeCV5.append(end - start)
  #   print("TimeCV5 = ",end - start )

  #   start = time.time()
  #   CrossValidateKNN(CV10,10,k)
  #   end = time.time()
  #   TimeCV10.append(end - start)
  #   print("TimeCV10 = ",end - start )

  #   start = time.time()
  #   CrossValidateKNN(CV15,15,k)
  #   end = time.time()
  #   TimeCV15.append(end - start)
  #   print("TimeCV15 = ",end - start )
  # plt.plot(K_Iteration, TimeTT7)
  # plt.plot(K_Iteration, TimeTT6)
  # plt.plot(K_Iteration, TimeTT5)

  # plt.plot(K_Iteration, TimeCV5)
  # plt.plot(K_Iteration, TimeCV10)
  # plt.plot(K_Iteration, TimeCV15)

  # plt.ylabel('Time')
  # plt.xlabel('K')
  # plt.legend(['Train&Test 0.7-0.3','Train&Test 0.6-0.4', 'Train&Test 0.5-0.5',
  #   'CrossValid 5Fold','CrossValid 10Fold','CrossValid 15Fold'])
  # plt.show()
  
knnMain('abalone.data', 0,0)
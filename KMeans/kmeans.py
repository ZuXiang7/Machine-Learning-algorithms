import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def loadData(filename):
  # load data from filename into X
  X=[]
  count = 0
  
  text_file = open(filename, "r")
  lines = text_file.readlines()
    
  for line in lines:
    X.append([])
    words = line.split("\t") #data values seperated by tab 
    # convert value of first attribute into float  
    for word in words:
      X[count].append(float(word))
    count += 1
  return np.asarray(X)

def errCompute(X, M):
  Summation = 0 #store total error
  k = M.shape[0]
  for clusterIndex in range(k): #get summation of distance for each index
    pointsInCluster = X[X[:,2] == clusterIndex, 0:2] #array of all points in this cluster
    currentMedoid = M[clusterIndex: clusterIndex + 1, :] #get median for this cluster
    medoidArray = np.repeat(currentMedoid, pointsInCluster.shape[0], axis = 0) #repeat so can minus all in one shot

    #find distance
    DistArr = pointsInCluster - medoidArray
    DistArr = np.square(DistArr)
    sumedArr = np.sum(DistArr,axis = 1)
    sqrtArr = np.sqrt(sumedArr)

    #summation of all in that cluster
    sqrtArr = np.sum(sqrtArr)
    Summation += sqrtArr

  #divide by n where n is number of samples  
  return Summation / X.shape[0]

def Group(X, M):
  k = M.shape[0]
  XCopy = np.copy(X)
  #copy just the samples
  Xpoints = np.copy(X[:, 0:2])
  

  #repeat each sample k times
  RepeatedX = np.repeat(Xpoints,k, axis = 0)

  #tile it 'number of sample' times so i can minus with repeated X
  TileM = np.tile(M,(X.shape[0],1))

  #distance of each sample with each medoid now clumped next to each other
  DistArr = RepeatedX - TileM
  DistArr = np.square(DistArr)
  sumedArr = np.sum(DistArr,axis = 1)
  sqrtArr = np.sqrt(sumedArr)

  #for each set of samples distance to medoid, find lowest and set to X column 3
  for i in range(X.shape[0]):
    XCopy[i][2] = np.argmin(sqrtArr[i * k: i * k + k])
  return XCopy

def calcMeans(X,M):
  k = M.shape[0]
  newM = np.copy(M)
  for clusterIndex in range(k):
    #get all samples from the current cluster
    pointsInCluster = X[X[:,2] == clusterIndex, 0:2]
    #get new mean
    mean = np.mean(pointsInCluster,axis = 0)
    #set mean to copy
    newM[clusterIndex: clusterIndex + 1, :] = mean
  return newM

Errors = []
Ks = []
def Main(filename, k):
  #data load
  X = loadData(filename)
  Ks.append(k)
  #plt data for part b
  # plt.plot(X[:,0:1],  X[:,1:2], 'ob')
  # plt.title('Section 1 Lightning data')
  # plt.show()

  #add one column for result
  zeros = np.zeros((X.shape[0], 1))
  X = np.hstack((X,zeros))

  M = np.copy(X[0:k, 0:X.shape[1]-1])
  
  prevX = np.ones((X.shape[0],1))
  while(True):
    prevX = np.copy(X[:,2:3])
    X = Group(X,M)
    if(np.array_equal(prevX , X[:,2:3])):
      break
    M = calcMeans(X,M)
  Errors.append(errCompute(X,M))
  print("for K = ",k,"Error = ",errCompute(X,M))
  # for clusterIndex in range(k):
  #   pointsInCluster = X[X[:,2] == clusterIndex, 0:2]
  #   plt.plot(pointsInCluster[:,0:1],  pointsInCluster[:,1:2],'o')
  # plt.title('Section 5, k = 100')
  # plt.show()
Main('2010825.txt', 5)
Main('2010825.txt', 50)
Main('2010825.txt', 100)
# plt.plot(Ks,  Errors)
# plt.ylabel('Error')
# plt.xlabel('K')
# plt.title('Error for each K')
# plt.show()
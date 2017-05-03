# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

## Submitted by Sai Nikhil Gundu, Id: 800962726

#groupId = org.apache.spark
#artifactId = spark-core_2.11
#version = 2.1.0

#groupId = org.apache.hadoop
#artifactId = hadoop-client
#version = Hadoop 2.6.0-cdh5.8.0

# Code developed using the skeleton code provided as base


import sys
import numpy as np

from pyspark import SparkContext

#Set the precision value for the output. 
np.set_printoptions(precision=13)

# defining the keys for computing beta values. (using the class lecture notes)

#######ALS User matrix row calculation
#Users[i] = inverse(Items*Items^T + I*lambda) * Items * A[i]^T
#Where only rows of item matrix for which user i has ratings are operated upon.

def Update_User(userTuple):
    '''
    This function calculates (userID, Users[i]) using:
        'Users[i] = inverse(Items*Items^T + I*lambda) * Items * A[i]^T'
    Dot product calculations are done differently than normal to allow for sparsity. Rather 
    than row of left matrix times column of right matrix, sum result of column of left matrix  
    * rows of right matrix (skipping items for which user doesn't have a rating).
    '''
    Itemssquare = np.zeros([n_factors.value,n_factors.value])
    for matrixA_item_Tuple in userTuple[1]:
        itemRow = Items_broadcast.value[matrixA_item_Tuple[0]][0]
        for i in range(n_factors.value):
            for j in range(n_factors.value):
                Itemssquare[i,j] += float(itemRow[i]) * float(itemRow[j])
    leftMatrix = np.linalg.inv(Itemssquare + lambda_.value * np.eye(n_factors.value))
    rightMatrix = np.zeros([1,n_factors.value])
    for matrixA_item_Tuple in userTuple[1]:
        for i in range(n_factors.value):
            rightMatrix[0][i] += Items_broadcast.value[matrixA_item_Tuple[0]][0][i] * float(matrixA_item_Tuple[1])
    newUserRow = np.dot(leftMatrix, rightMatrix.T).T
    return (userTuple[0], newUserRow)



#######ALS Item matrix row calculation
#Items[i] = inverse(Users*Users^T + I*lambda) * Users * A[i]^T
#Where only rows of item matrix for which user i has ratings are operated upon.

def Update_Item(userTuple):
    '''
    This function calculates (userID, Items[i]) using:
        'Items[i] = inverse(Users*Users^T + I*lambda) * Users * A[i]^T'
    Dot product calculations are done differently than normal to allow for sparsity. Rather 
    than row of left matrix times column of right matrix, sum result of column of left matrix  
    * rows of right matrix (skipping items for which user doesn't have a rating).
    '''
    Userssquare = np.zeros([n_factors.value,n_factors.value])
    for matrixA_user_Tuple in userTuple[1]:
        userRow = Users_broadcast.value[matrixA_user_Tuple[0]][0]
        for i in range(n_factors.value):
            for j in range(n_factors.value):
                Userssquare[i,j] += float(userRow[i]) * float(userRow[j])
    leftMatrix = np.linalg.inv(Userssquare + lambda_.value * np.eye(n_factors.value))
    rightMatrix = np.zeros([1,n_factors.value])
    for matrixA_user_Tuple in userTuple[1]:
        for i in range(n_factors.value):
            rightMatrix[0][i] += Users_broadcast.value[matrixA_user_Tuple[0]][0][i] * float(matrixA_user_Tuple[1])
    newItemRow = np.dot(leftMatrix, rightMatrix.T).T
    return (userTuple[0], newItemRow)

def getRowSumSquares(userTuple):
    userRow = Users_broadcast.value[userTuple[0]]
    rowSSE = 0.0
    for matrixA_item_Tuple in userTuple[1]:
        predictedRating = 0.0
        for i in range(n_factors.value):
            predictedRating += userRow[0][i] * Items_broadcast.value[matrixA_item_Tuple[0]][0][i]
        SE = (float(matrixA_item_Tuple[1]) - predictedRating) ** 2
        rowSSE += SE
    return rowSSE



if __name__ == "__main__":
  
  sc = SparkContext(appName="Custom_ALS")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  ratings_raw_data = sc.textFile(sys.argv[1])
  ratings_raw_data_header = ratings_raw_data.take(1)[0]

  movies_raw_data = sc.textFile(sys.argv[2])
  movies_raw_data_header = movies_raw_data.take(1)[0]
  #yxinputFile = sc.textFile(sys.argv[1])
  #yxlines = yxinputFile.map(lambda line: line.split(','))
  
  ratings_data = ratings_raw_data.filter(lambda line: line!=ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
  ratings_1_data_forM_U = ratings_raw_data.filter(lambda line: line!=ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[1],tokens[0],tokens[2])).cache()
    
  

  movies_data = movies_raw_data.filter(lambda line: line!=movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
  
  #Calculating training and test splits-Using entire data as training
  
  training_RDD, test_RDD = ratings_data.randomSplit([10,0], seed=9)
  training_RDD1,test_RDD1 = ratings_1_data_forM_U.randomSplit([10,0], seed=9)
  
  # Creating sparse representation of A matrix with users as rows and items as columns

  user_item_ratings = (training_RDD
                          .map(lambda p: (p[0],p[-2:])).groupByKey()).cache()
  
  item_user_ratings=(training_RDD1
                          .map(lambda p: (p[0],p[-2:])).groupByKey()).cache()
    
    
  # User Defined Parameters
  lambda_ = sc.broadcast(0.1) # Regularization parameter
  n_factors = sc.broadcast(3) # nfactors of User matrix and Item matrix
  n_iterations = 20 # How many times to iterate over the user and item matrix calculations.
  
  # Initizializing Items Matrix (User matrix doesn't need to be initialized since it is solved for first):
  Items=dict()
  Items = item_user_ratings.map(lambda line: (line[0], 5 * np.random.rand(1, n_factors.value)))
    
  # The item matrix is needed in all partitions when solving for rows of User matrix individually
  # The item matrix is needed in all partitions when solving for rows of User matrix individually
  Items_broadcast = sc.broadcast({
    k: v for (k, v) in Items.collect()
  })



  from pandas import *
  from itertools import *
  j = 0
  for k, v in {k: v for (k, v) in Items.collect()}.items():
      print k, v
      j+=1
      if j > 10:
          break
        
   
  j = 0
  for i in user_item_ratings.take(1)[0][1]:
      print(i)
      j+=1
      if j > 10:
          break
        
        
  Users = user_item_ratings.map(Update_User)
  Items = item_user_ratings.map(Update_Item)
  print "Matrix Factorization completed successfully !"

  # The is needed in all partitions when solving for rankings
  Users_broadcast = sc.broadcast({
    k: v for (k, v) in Users.collect()
  })
  
    
  # This is needed in all partitions when solving for rankings
  Items_broadcast = sc.broadcast({
    k: v for (k, v) in Items.collect()
  })

    
  SSE = user_item_ratings.map(getRowSumSquares).reduce(lambda a, b: a + b)
  Count = ratings_data.count()
  MSE = SSE / Count
  print ("MSE:", MSE)
    
  #userRow=Users_broadcast.value[userId][0]
      
  item_keys= Items.map(lambda x: (x[0])).collect()

  print "####################### ALS model built successfully ####################"
 
  print "-------------------------------------------------------------------------"
  
  print "-------------------------------------------------------------------------"

  print "Now Recommending movies for the user"
  
  userId = (sys.argv[3])

  k=Users_broadcast.value[userId][0]
    
  dim=item_user_ratings.count()

  
  #Initialize the user prediction matrix
  u=[0.00000]*9066
  user_pred=np.array(u)

  for i, j in zip(range(dim),item_keys):
      itemRow=Items_broadcast.value[j]
      user_pred[i]=(np.dot(k,itemRow.T))
  
  #Printing movie titles
  # The item matrix is needed in all partitions when solving for rows of User matrix individually
  movies_broadcast = sc.broadcast({
    k: v for (k, v) in movies_data.collect()
  })
    
  top_pred_keys=user_pred.argsort()[-5:][::-1]
  
  top_movie_keys=np.array([0]*5)
  print("Recommended movies for userId", userId)
  for i, j in zip(range(5),top_pred_keys):
      print "Movie Id=", item_keys[j], " with title=", movies_broadcast.value[item_keys[j]] 
  


  sc.stop()

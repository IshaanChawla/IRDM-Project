import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import threading
import time
import os

class KMeans:
    def __init__(self,inputFile,k = -1,iterations = 10):
        ''' Initializing the Classifier '''

        # Reading the input data from the .csv file and creating a Pandas DataFrame
        self.__inputDataFrame = pd.read_csv('./inputs/' + inputFile)
        # No of Training Examples
        self.__m = self.__inputDataFrame.shape[0]
        # No of features
        self.__n = self.__inputDataFrame.shape[1]

        # self.__predefinedK stores if the user wants to number of clusters to predefined or not
        self.__predefinedK = True if(k != -1) else False
        # self.k saves the number of clusters formed
        self.__k = 1 if(k == -1) else k

        # Taking Max and Min Value of all features to make random initializations
        self.__maxFeatures = self.__inputDataFrame.max()
        self.__minFeatures = self.__inputDataFrame.min()

        # self.__iter saves the number of random initiations for a value of k user wants
        self.__iter = iterations

        # self.__class stores the classes of the Training Examples
        self.__output = {}

        # self.__outputPath saves the path where graphs and file containing the centroids are saved
        self.__outputPath = './outputs/' + inputFile.split(".")[0] + '/'
        if not os.path.exists(self.__outputPath):
            os.makedirs(self.__outputPath)

    def __initializeCentroids(self,k):
        ''' Initialize the Centroids and return Panda DataFrame '''
        centroids = []
        probab = [0 for i in range(self.__m)]
        # Initializing the first centroid as a random point of the data
        firstCentroid = self.__inputDataFrame.iloc[random.randint(0, self.__m - 1)]
        centroids.append([firstCentroid[j] for j in range(self.__n)])
        # Initializing more centroids based on Maximum distance
        for j in range(k-1):
            allDistanceList = self.__calculateCentroidsDistance(pd.DataFrame(centroids, columns = self.__inputDataFrame.columns.values),j+1)
            for i in range(self.__m):
                probab[i] = min([allDistanceList[m][i] for m in range(j+1)])

            nextCentroid = self.__inputDataFrame.iloc[probab.index(max(probab))]
            centroids.append([nextCentroid[j] for j in range(self.__n)])
        return pd.DataFrame(centroids, columns = self.__inputDataFrame.columns.values)

    def __calculateCentroidsDistanceThreads(self,centroids,j,distanceList):
        ''' Threads To make Calculations Faster '''
        # Looping over all Training Examples
        for i in range(self.__m):
            # Distance from the centroid is initialized to be Zero
            distance = 0
            # Looping over all Feature Values and Calculating Euclidean Distance
            for n in range(self.__n):
                distance += pow(self.__inputDataFrame.iloc[i][n] - centroids.iloc[j][n], 2)
            distanceList.append(distance)

    def __calculateCentroidsDistance(self,centroids,k):
        ''' Function to calculate Distance of Points from Centroids by Launching Threads '''
        # Looping over all Centroids
        allDistanceList = []
        threads = []
        for j in range(k):
            distanceList = []
            t = threading.Thread(target=self.__calculateCentroidsDistanceThreads, args=(centroids, j, distanceList,))
            t.start()
            threads.append(t)
            allDistanceList.append(distanceList)
        # Waiting Until all Threads Join
        for j in range(k):
            threads[j].join()
        return allDistanceList

    def __assignClusters(self,centroids,k):
        ''' Function which assigns the cluster to every Training Example by allocating the class of the nearest centroid '''
        classified = np.array([-1 for i in range(self.__m)])
        allDistanceList = self.__calculateCentroidsDistance(centroids,k)
        # Assigning Cluster to the Training Examples
        for i in range(self.__m):
            values = [allDistanceList[j][i] for j in range(k)]
            classified[i] = values.index(min(values))
        return classified

    def __recomputeCentroidsThread(self,centroids,classified,j):
        ''' Threads To make Calculations Faster '''
        # Calculating all the Training Examples which are classified under the class of this centroid
        indeces = np.where(classified == j)[0]
        # Looping over all the features of the centroids and recomputing its coordinates with mean of all Training Examples
        for colName in self.__inputDataFrame.columns.values:
            centroids.iloc[j][colName] = self.__inputDataFrame.iloc[indeces][colName].mean()

    def __recomputeCentroids(self,centroids,k,classified):
        ''' Reassigning the coordinates to the Centroids of the clusters '''
        threads = []
        # Looping over all the centroids
        for j in range(k):
            t = threading.Thread(target=self.__recomputeCentroidsThread, args=(centroids, classified, j,))
            t.start()
            threads.append(t)

        # Waiting Until all Threads Join
        for j in range(k):
            threads[j].join()

    def __showGraph2D(self,centroids,classified,k):
        ''' Function to show K Mean for 2 Feature Data '''
        x,y = self.__inputDataFrame.columns.values
        plt.scatter(self.__inputDataFrame[x],self.__inputDataFrame[y],c = classified,marker = ".",s = 50)
        plt.scatter(centroids[x],centroids[y],c = [i for i in range(k)],marker = "*",s = 70)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title('K Means Graph')
        plt.savefig(self.__outputPath + '2D Clustered Data.png', bbox_inches='tight')
        plt.close()

    def __showGraph3D(self,centroids,classified,k):
        ''' Function to show K Mean for 3 Feature Data '''
        x,y,z = self.__inputDataFrame.columns.values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.__inputDataFrame[x],self.__inputDataFrame[y],self.__inputDataFrame[z],c = classified,marker = ".",s = 50)
        ax.scatter(centroids[x],centroids[y],centroids[z],c = [i for i in range(k)],marker = "*",s = 70)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.savefig(self.__outputPath + '3D Clustered Data.png',bbox_inches='tight')
        plt.close()

    def __showGraphObjectiveFuntion2D(self):
        ''' Function to show K v/s Objective Function Value '''
        x = [key for key in self.__output]
        y = [math.log(value[1],10) for key,value in self.__output.items()]

        # Calculating the slopes by taking difference in y as change in x is a unit
        slopes = [j-i for i, j in zip(y[:-1], y[1:])]
        # deltaSlopes save the values of change in slope to find the Elbow Point
        deltaSlopes = [j-i for i, j in zip(slopes[:-1], slopes[1:])]
        self.__elbowPoint = deltaSlopes.index(max(deltaSlopes)) + 2
        plt.scatter(x, y)
        plt.plot(x, y)
        plt.xlabel('K')
        plt.ylabel('log(Objective Function Value)')
        plt.savefig(self.__outputPath + 'Objective Function vs K.png', bbox_inches='tight')
        plt.close()


    def __computeObjectiveFunction(self,centroids,classified):
        ''' Function computes the value of the objective function '''
        rssFunction = 0
        for i in range(self.__m):
            distance = 0
            for j in range(self.__n):
                distance += pow(self.__inputDataFrame.iloc[i][j]-centroids.iloc[classified[i]][j],2)
            rssFunction += pow(distance,0.5)
        return rssFunction

    def __runIteration(self,k):
        ''' Function which runs the K-Means Algo '''
        centroids = self.__initializeCentroids(k)
        # lastClassified tells if there are any changes in classification or not
        lastClassified = np.array([[-1 for i in range(self.__m)]])
        classified = np.array([])
        while True:
            classified = self.__assignClusters(centroids,k)
            # If the current classification is equal to the previous classification then clusters have been formed. Stop Calculations
            if (lastClassified == classified).all():
                break
            self.__recomputeCentroids(centroids,k,classified)
            lastClassified = classified

        # Calculating the Objective Function to check that
        rssFunction = self.__computeObjectiveFunction(centroids,classified)

        # Insert in output Dictionary if key doesnt exist or Update the Output Dictionary if Objective Function decreases
        if (k not in self.__output.keys() or self.__output[k][1] > rssFunction):
            self.__output[k] = (centroids, rssFunction, classified)

    def __iterationHandler(self):
        ''' Function to Handle all iterations over a single value of K '''
        threads = []
        # Starting all iterations with Random Iterations Simultaneously
        for i in range(self.__iter):
            t = threading.Thread(target = self.__runIteration,args = (self.__k,))
            t.start()
            threads.append(t)

        # Waiting For the Threads to end
        for i in range(self.__iter):
            threads[i].join()

    def classify(self):
        ''' Function which acts as an interface between the user and the class '''
        if self.__predefinedK:
            self.__iterationHandler()
            self.__elbowPoint = self.__k
        else:
            # If K is not Predefined then make the iterations until K is less than 2*root(Training Examples / 2). Just an Heuristic
            while self.__k <= 15:
                print(self.__k)
                self.__iterationHandler()
                self.__k += 1
            self.__showGraphObjectiveFuntion2D()

        # Write Centroid of the Elbow Point to the CSV
        self.__output[self.__elbowPoint][0].to_csv(self.__outputPath + 'centroid.csv',index=False)

        # Show the graph changing if no of features are 2 or 3
        if self.__n == 2:
            self.__showGraph2D(self.__output[self.__elbowPoint][0],self.__output[self.__elbowPoint][2],self.__elbowPoint)
        if self.__n == 3:
            self.__showGraph3D(self.__output[self.__elbowPoint][0],self.__output[self.__elbowPoint][2],self.__elbowPoint)


if __name__ == '__main__':
    start_time = time.time()
    kMeansClassifier = KMeans('House Images.csv',iterations = 3)
    kMeansClassifier.classify()
    print(str(time.time() - start_time) + " sec")
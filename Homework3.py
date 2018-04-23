# CSCI 3302: Homework 3 -- Clustering and Classification
# Implementations of K-Means clustering and K-Nearest Neighbor classification

# Sherry Nguyen

import pickle
import math 
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

class KMeansClassifier(object):

  def __init__(self):
    self._cluster_centers = [] # List of points representing cluster centers
    self._data = [] # List of datapoints (lists of real values)

  def add_datapoint(self, datapoint):
    self._data.append(datapoint)

  def fit(self, k):
    #print self._data
    # Fit k clusters to the data, by starting with k randomly selected cluster centers.
    # HINT: To choose reasonable initial cluster centers, you can set them to be in the same spot as random (different) points from the dataset
    self._cluster_centers = []
    temp = []#list of temporary centers
    length = len(self._data) 
    for x in range(k):
      y = random.randint(0, length-1)
      print y
      temp.append(self._data[y])
      #print self._data[y]
    #print temp

    # TODO Follow convergence procedure to find final locations for each center
    merp = False
    tolerance = 0.1
    while (merp != True):
      #print temp
      x_data = [i[0] for i in self._data]
      y_data = [i[1] for i in self._data]
      #print x_data, y_data
      x_cluster = [j[0] for j in temp]
      y_cluster = [j[1] for j in temp]
      #print x_cluster

      #Find closest cluster and append to list 
      for a in range(length):
        #print self._data[a]
        distance = 5000
        for b in range (k):
          c = x_cluster[b]-x_data[a]
          d = y_cluster[b]-y_data[a]
          c = math.pow(c, 2)
          d = math.pow(d, 2)
          dist = math.sqrt(c+d)
          if(dist == 0):
            b = k + 1
          #print dist
          elif (dist < distance):
            cluster = b
            distance = dist
        if (dist != 0):
          c = self._data[a]
          temp[cluster].extend(c)

      #Find new centers
      new_temp = []
      for h in range(k):
        sumx = 0
        sumy = 0
        for g in xrange(0, len(temp[h]), 2):
            sumx += temp[h][g]
            sumy += temp[h][g+1]
        #print sumx
        #print sumy
        new_x = sumx/((len(temp[h])/2))
        new_y = sumy/((len(temp[h])/2))
        new_temp.append([new_x, new_y])
      #print new_temp
      #print new_temp[0], temp[0]
      
      #Check difference from centers to new centers
      isTrue = 0
      for w in range(k):
        diff_x = math.fabs(temp[w][0] - new_temp[w][0])
        diff_y = math.fabs(temp[w][1] - new_temp[w][1])
        #print diff_x, diff_y
        if (diff_x < tolerance and diff_y < tolerance):
          isTrue += 1          
      temp = new_temp
      if isTrue == k:
        merp = True
      #print temp
      #print self._data

    # TODO Add each of the 'k' final cluster_centers to the model (self._cluster_centers)
    self._cluster_centers = temp

  def classify(self,p):
    #print self._data
    # Given a data point p, figure out which cluster it belongs to and return that cluster's ID (its index in self._cluster_centers)
    closest_cluster_index = None
    print 'Clusters:', self._cluster_centers
    # TODO Find nearest cluster center, then return its index in self._cluster_centers
    x_data = self._data[p][0]
    y_data = self._data[p][1]
    print x_data, y_data
    x_cluster = [j[0] for j in self._cluster_centers]
    y_cluster = [j[1] for j in self._cluster_centers]
    distance = 500
    print 'Distances from Clusters:'
    for b in range (len(self._cluster_centers)):
      c = x_cluster[b]-x_data
      d = y_cluster[b]-y_data
      c = math.pow(c, 2)
      d = math.pow(d, 2)
      dist = math.sqrt(c+d)
      print dist
      if (dist <= distance):
        cluster = b
        distance = dist
    closest_cluster_index = cluster
    print 'Closest Cluster:'
    print closest_cluster_index
    #print self._data
    
    #Gives visual of cluster centers and the point being tested
    #x_val = [x[0] for x in self._cluster_centers]
    #y_val = [x[1] for x in self._cluster_centers]
    #x_closest = self._cluster_centers[cluster][0] 
    #y_closest = self._cluster_centers[cluster][1]
    #x_merp = [x[0] for x in self._data]
    #y_merp = [x[1] for x in self._data]
    #plt.scatter(x_merp, y_merp, c='r')
    #plt.scatter(x_val, y_val, c='g')
    #plt.scatter(x_closest, y_closest, c='b')
    #plt.scatter(x_data, y_data)
    #plt.show()
    
    return closest_cluster_index

class KNNClassifier(object):

  def __init__(self):
    self._data = [] # list of (data, label) tuples
  
  def clear_data(self):
    self._data = []

  def add_labeled_datapoint(self, data_point, label):
    self._data.append((data_point, label))
    #print self._data
  
  def classify_datapoint(self, k, data_point):
    label_counts = {} # Dictionary mapping "label" => count
    best_label = None
  
    #TODO: Perform k_nearest_neighbor classification, set best_label to majority label for k-nearest points
    #print data_point
    distance = [500]*k
    index = np.zeros(k)#keep track of which point in self._data is close to the point
    #print distance
    #print data_point
    x_val = data_point[0]
    y_val = data_point[1]
    #print x_val, y_val
    merp = [w[0] for w in self._data]
    x_data = [x[0] for x in merp]
    y_data = [x[1] for x in merp]
    for i in range(0, len(self._data)):
      diff_x = x_data[i] - x_val
      diff_y = y_data[i] - y_val
      c = math.pow(diff_x, 2)
      d = math.pow(diff_y, 2)
      dist = math.sqrt(c+d)
      #print dist, distance[k-1]
      if dist <= distance[k-1]:
        for j in range(k):
          if dist <= distance[j]:
            #print j
            inx = j
            break
        #print inx      
        for a in range(k-1, inx, -1):
          distance[a] = distance[a-1]
        for b in range(k-1, inx, -1):
          index[b] = index[b-1]
        distance[inx] = dist
        index[inx] = i
        #print distance, index         
          
    label = []
    for l in range(k):
      merr = int(index[l])
      #print merr, self._data[merr][1]
      argh = self._data[merr][1]
      #print argh
      label.append(argh)
    #print label
    #print most_common(label)
    most_common_label = [word for word, word_count in Counter(label).most_common(1)]
    #print most_common_label
    labels = most_common_label[0]
    #print labels
    best_label = labels

    return best_label



def print_and_save_cluster_centers(classifier, filename):
  for idx, center in enumerate(classifier._cluster_centers):
    print "  Cluster %d, center at: %s" % (idx, str(center))


  f = open(filename,'w')
  pickle.dump(classifier._cluster_centers, f)
  f.close()

def read_data_file(filename):
  f = open(filename)
  data_dict = pickle.load(f)
  f.close()

  return data_dict['data'], data_dict['labels']


def main():
  # read data file
  data, labels = read_data_file('hw3_data.pkl')
  #print data
  #print len(data)

  #x_val = [x[0] for x in data]
  #y_val = [x[1] for x in data]
  #plt.scatter(x_val, y_val)
  #plt.show()

  # data is an 'N' x 'M' matrix, where N=number of examples and M=number of dimensions per example
  # data[0] retrieves the 0th example, a list with 'M' elements
  # labels is an 'N'-element list, where labels[0] is the label for the datapoint at data[0]


  ########## PART 1 ############
  # perform K-means clustering
  kMeans_classifier = KMeansClassifier()
  for datapoint in data:
    kMeans_classifier.add_datapoint(datapoint) # add data to the model

  kMeans_classifier.fit(4) # Fit 4 clusters to the data
  kMeans_classifier.classify(8) # Classifying certain data point to clusters, you can change the point if you want

  # plot results
  print '\n'*2
  print "K-means Classifier Test"
  print '-'*40
  print "Cluster center locations:"
  print_and_save_cluster_centers(kMeans_classifier, "hw3_kmeans_Sherry.pkl")

  print '\n'*2


  ########## PART 2 ############
  print "K-Nearest Neighbor Classifier Test"
  print '-'*40

  # Create and test K-nearest neighbor classifier
  kNN_classifier = KNNClassifier()
  k = 2
  #print labels
  correct_classifications = 0
  #Perform leave-one-out cross validation (LOOCV) to evaluate KNN performance
  #print data
  #for holdout_idx in range(len(data)):
  for holdout_idx in range(len(data)):
    #print holdout_idx
    # Reset classifier
    kNN_classifier.clear_data()

    for idx in range(len(data)):
      #print idx
      if idx == holdout_idx: continue # Skip held-out data point being classified
      # Add (data point, label) tuples to KNNClassifier
      kNN_classifier.add_labeled_datapoint(data[idx], labels[idx])
    # changed it from passing (data[holdout_idx], k) to what it is now
    guess = kNN_classifier.classify_datapoint(k, data[holdout_idx]) # Perform kNN classification
    #print guess, labels[holdout_idx]
    if guess == labels[holdout_idx]: 
      correct_classifications += 1.0
  #print labels
  print "kNN classifier for k=%d" % k
  print "Accuracy: %g" % (correct_classifications / len(data))
  print '\n'*2



if __name__ == '__main__':
  main()

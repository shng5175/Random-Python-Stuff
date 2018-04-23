
###############################################################################
#Question 1

import random
import numpy as np

print 'Question 1:'
a = []
size = 100
for x in range(size):
    y = random.randint(0, size)
    a.append(y)
print 'Array of numbers:'
print a
max_reverses = 0
for z in range(size):
    max_reverses += size - z -1
#print max_reverses
probability = 0.5
probability_reverse = max_reverses * 0.5
#print probability_reverse

#Solving in n^2 time
#Using this to help with comparison since I know this is accurate
count_1 = 0
for i in range(len(a)):
    for j in range(i, len(a)):
        if (a[i] > a[j]):
            count_1 += 1
print 'Anwer we should get: ', count_1

###############################################################################
#Solving in O(nlogn) time using mergeSort
#Basic mergeSort function
def mergeSort(arr, temp, left, right):
    #print 'Merp'
    count = 0
    if left < right:
        middle = int((left + right)/2)
        count += mergeSort(arr, temp, left, middle)
        count += mergeSort(arr, temp, middle + 1, right)
        count += merge(arr, temp, left, middle, right)
    return count

def merge(arr, temp, left, middle, right):
    #print left, middle, right
    count = 0
    i = left #i is index for left subarray
    j = left #j is index for the result of the merged subarray
    k = middle + 1 #index for the right subarray
    while ((i <= middle) and (k <= right)):
        #print 'FOOOD'
        #compare smallest elements of each subarray
        #if the element at i is larger that that at k, then k is smaller than all the elments in the left subarray
        if (arr[i] > arr[k]):
            count += ((middle + 1) - i)#increase count by the amount of numbers in the left subarray that are larger than the element at k
            temp[j] = arr[k] #since we now know the elment at k is amller than everything else we can append it to the temporary array
            j += 1
            k += 1
        #since the element at i isn't larger, we can now append it to the temporary list
        else:
            temp[j] = arr[i]        
            j += 1
            i += 1            

    #Append all remaining elements in the left subarray into the temporary array
    while(i <= middle):
        #print i
        temp[j] = arr[i]
        j += 1
        i += 1        

    #Append all remaining elements in the right subarray into the temporary array
    while(k <= right):
        #print k, right
        temp[j] = arr[k]
        j += 1
        k += 1

    #We are now done with the original array, can now copy everything in temp to the original array
    for l in range(left, right + 1):
        arr[l] = temp[l]

    return count

length = len(a)
temp = np.zeros(length)
print 'Answer: ', mergeSort(a, temp, 0, length - 1)

print'\n\n\n\n'

###############################################################################
#Question 4

print 'Question 4:'
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#create a directed graph
G = nx.DiGraph()

#adding an edge also adds the node
G.add_edge('Spider', 'A', weight=1.0)
G.add_edge('Spider', 'H', weight=1.0)
G.add_edge('Spider', 'J', weight=1.0)

G.add_edge('H', 'G', weight=1.0)
G.add_edge('H', 'K', weight=1.0)

G.add_edge('G', 'L', weight=1.0)
G.add_edge('G', 'F', weight=1.0)

G.add_edge('F', 'E', weight=1.0)

G.add_edge('E', 'Fly', weight=1.0)

G.add_edge('J', 'S', weight=1.0)
G.add_edge('J', 'K', weight=1.0)

G.add_edge('K', 'L', weight=1.0)
G.add_edge('L', 'M', weight=1.0)
G.add_edge('M', 'N', weight=1.0)
G.add_edge('M', 'F', weight=1.0)

G.add_edge('N', 'O', weight=1.0)
G.add_edge('N', 'E', weight=1.0)

G.add_edge('O', 'Fly', weight=1.0)

G.add_edge('A', 'S', weight=1.0)
G.add_edge('A', 'B', weight=1.0)

G.add_edge('B', 'R', weight=1.0)
G.add_edge('B', 'C', weight=1.0)

G.add_edge('S', 'R', weight=1.0)
G.add_edge('R', 'Q', weight=1.0)

G.add_edge('Q', 'C', weight=1.0)
G.add_edge('Q', 'P', weight=1.0)

G.add_edge('C', 'D', weight=1.0)
G.add_edge('D', 'Fly', weight=1.0)
G.add_edge('P', 'D', weight=1.0)
G.add_edge('P', 'O', weight=1.0)
G.add_edge('O', 'Fly', weight=1.0)

G.add_edge('T', 'Q', weight=1.0)
G.add_edge('T', 'P', weight=1.0)
G.add_edge('T', 'O', weight=1.0)
G.add_edge('T', 'N', weight=1.0)
G.add_edge('T', 'M', weight=1.0)

G.add_edge('R', 'T', weight=1.0)
G.add_edge('S', 'T', weight=1.0)
G.add_edge('J', 'T', weight=1.0)
G.add_edge('K', 'T', weight=1.0)
G.add_edge('L', 'T', weight=1.0)

#each edge has a weight of 1. The shortest path is the fewest edges.
#Use this to verify that your graph built correctly.
t = nx.shortest_path(G, 'Spider', 'Fly', weight='weight')

#print(t)

#Tried to get a visual of the graph
#val_map = {'Spider': 1.0,
#           'J': 0.2,
#           'T': 0.4,
#           'O' : 0.6,
#           'Fly': 0.0}
#values = [val_map.get(node, 0.25) for node in G.nodes()]
#nx.draw(G, cmap = plt.get_cmap('jet'), node_color = values)
#plt.show()

#Messing around with networkx and learning about the built-in functions that might be useful
#print G.node()
#print G.edges()
#print G.edges('Spider')
#print G.in_edges('Fly')
#print G.has_edge('O', 'Fly')
#print len(G)

###############################################################################
#This is where I started implementing my algorithm
length = len(G)#number of nodes in G
nodes = ['Fly']#list of nodes to be traversed starting with the node 'Fly'
#print len(G.in_edges('Fly'))

#Find order in which nodes should be traversed
#There's probably a better way to do this, but if it works, that's all that matters to me right now
##for j in range(length):
##    for i in G.nodes:
##        #if there is an edge between i to any node in j so far
##        if (G.has_edge(i, nodes[j])):
##            print j, nodes[j], i
##            for k in range(len(nodes)):
##                print 'kjakljflsajfljslkvdjlsavjls', k, nodes[k], i
##                if nodes[k] == i:# if node already exists in the array
##                    print nodes
##                    print k, nodes[k], i
##                    nodes.pop(k)#pop it out since it is dependent on an element after it
##                    if (k <= j):
##                        j = j-1
##                    break
##            nodes.append(i)#append node i to array
##    print nodes
##    print j
##print nodes   
###print len(nodes)
##if(len(nodes)!= len(G)):
##    print "ERROR"

#guess my algorithm actually starts here, sorry about that
#Part of the list this generates isn't quite correct,
#For some reason doesn't really iterate through j = 14
#I'll try working on it some more if I have time
#Should be something like:
node_list = ['Fly', 'E', 'D', 'O', 'F', 'C', 'P', 'N', 'Q', 'M',
             'T', 'R', 'L', 'K', 'S', 'B', 'G', 'H', 'A', 'J', 'Spider']
path_count = np.zeros(len(node_list))#number of paths from 'Fly' to itself should be 0

for i in range(1, len(node_list)):
    if G.has_edge(node_list[i], node_list[0]):#if has edge to goal, increment number of paths for that node
        path_count[i] += 1
    #Check to see if has edge from any or the nodes that we've already solved for number of paths to the goal
    for j in range(i, 0, -1):
        #If there is an edge, then the number of paths from that node is equal to itself plus the number of paths it has an edge with
        if G.has_edge(node_list[i], node_list[j]):
            path_count[i] += path_count[j]
print 'List of nodes: '
print node_list
#print path_count # gets a visual of the number of paths at each node
print 'Total number of paths from Spider to Fly: ', int(path_count[i]), ' paths'

###############################################################################
#Answer I should get
paths = nx.all_simple_paths(G, source='Spider', target='Fly', cutoff=21)
#print len(list(paths))
if (int(path_count[i]) == len(list(paths))):
    print 'Correct!'

print'\n\n\n\n'

################################################################################
#Question 5

print 'Question 5: '
import numpy as np

#list of tuples containing a person, distance from closest partner, and their partner
data = [('a', 2, 'b'), ('b', 2, 'a'), ('c', 3, 'b')] 
#print data[1][2]
n = len(data)
hit = np.zeros((n), dtype = bool)
#print hit
for i in range(n):
    for j in range(i, n):
        if ((hit[i] == False) and (data[i][2] == data[j][0]) and (data[i][1] == data[j][1])):
            hit[i] = True
            hit[j] = True
#Show the array or whether or not someone was hit
#print hit

#if even one of the items in the array is False, then the question is False
for k in range(n):
    if (hit[k] == False):
        print 'False'

import numpy as np


#list of tuples containing a person, distance from partner, and their partner
data = [('a', 2, 'b'), ('b', 2, 'a'), ('c', 1, 'b')] 
#print data[1][2]
n = len(data)
hit = np.zeros((n), dtype = bool)
#print hit
for i in range(n):
    for j in range(i, n):
        if ((hit[i] == False) and (data[i][2] == data[j][0]) and (data[i][1] == data[j][1])):
            hit[i] = True
            hit[j] = True
print hit

for k in range(n):
    if (hit[k] == False):
        print 'False'
    





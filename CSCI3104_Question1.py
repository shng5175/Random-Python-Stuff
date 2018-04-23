import random
import numpy as np

a = []
size = 100
for x in range(size):
    y = random.randint(0, size)
    a.append(y)
print a

#Solving in n^2 time
#Using this to help with comparison since I know this is accurate
count_1 = 0
for i in range(len(a)):
    for j in range(i, len(a)):
        if (a[i] > a[j]):
            count_1 += 1
print count_1

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
print mergeSort(a, temp, 0, length - 1)

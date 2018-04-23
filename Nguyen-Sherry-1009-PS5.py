import random

f = file ("dist.all.last.txt", "r")
print f.read
#count = 0;
#for i in f:
#    count +=1
#print count
#had this to find out how many lines were in the file, turned out to be unecessary, but it got the correct number
names = []
length = 88799
for line in f:
    names.append(line)
#Check to see if names were in array
#print names[length-1]
f.close()
l = 5701
randomNames = []
half = length/2
for i in range(1, half):
    index = random.randint(0, len(names)-1)
    randomNames.append(names[index])
    names.pop(index)#takes it out of names so it can't be used again
#Hash 1 function
hash1 = []
for j in randomNames:
    k = 0
    for x in range(0, len(j)):
        k = k +(ord(j[x])-ord("A")+1)
        k = k % l
    hash1.append(k)
#ouputting to file to make it easier to make the histogram
file1 = open("Hash1_Results.txt", "w")
for a in hash1:
    file1.write("%s\n" % a)
file1.close()
#print(hash1)

#Hash 2 function
hash2 = []
for j in randomNames:
    k = 0
    for x in range(0, len(j)):
        y = (ord(j[x])-ord("A")+1)
        z = random.randint(0, 1)
        k = k + (y*z)
        k = k % l
    hash2.append(k)
#print(hash2)
file2 = open("Hash2_Results.txt", "w")
for b in hash2:
    file2.write("%s\n" % b)
file2.close()

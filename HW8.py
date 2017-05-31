import sys
import math

x = int(sys.argv[1]) #takes in the number as an arguement
y = math.sqrt(x) #gets the square root of x
y = math.ceil(y) #rounds up y
y = int(y) #did this because the for loop didn't want to take in a float as part of the range

i = 2
for i in range(i, y): #this whole section here finds prime numbers from 2 to the square root of x
   j = 2 
   while (j <= (i/j)): 
      if (i%j == 0): 
         break #sends to end of for loop where we increment i
      j = j + 1 #increments j and sends back top of loop to test i again
      if (j > i/j): #only ever gets to this line if i is indeed prime because there were no js that evenly goes into it
         if(x%i==0): #checks if the prime number we just found is a factor of the original number
            p = i #sets the number to p if the if statement above is true
            i = (y + 1) #since we've found p, sets i to y+1 to exit the loop
   i = i + 1
q=x/p #since we've found p, we can calculate the value of q
print "p = ",p , "q = ",q #prints out the values of p and q
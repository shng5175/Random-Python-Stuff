name = raw_input('What is your name?\n')
print 'Please input your scores for the following: \n'
a = input('Group quizzes: ')
b = input('You quizzes: ')
c = input('Prograaming and written assignments: ')
d = input('Exam 1: ')
e = input('Exam 2: ')
w = input('Final Exam: ')
textfile = raw_input("Enter the ouput file name: ")
a = a*0.14
b = b*0.14
c = c*0.14
d = d*0.13
e = e*0.20
w = w*0.25
g = (a+b+c+d+e+w)
print "Your course grade is: ", g
f = file( textfile, "a")
f.write ("\n")
f.write (name)
f.write(" ")
f.write ("%.2f" % a)
f.write(" ")
f.write ("%.2f" % b)
f.write(" ")
f.write ("%.2f" % c)
f.write(" ")
f.write ("%.2f" % d)
f.write(" ")
f.write ("%.2f" % e)
f.write(" ")
f.write ("%.2f" % w)
f.write(" ")
f.write ("%.2f" % g)
f.close()


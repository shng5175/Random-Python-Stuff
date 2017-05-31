name = raw_input('What is your name?\n')
print 'Please input your scores for the following: \n'
a = input('Group quizzes: ')
b = input('You quizzes: ')
c = input('Prograaming and written assignments: ')
d = input('Exam 1: ')
e = input('Exam 2: ')
f = input('Final Exam: ')
a = a*0.14
b = b*0.14
c = c*0.14
d = d*0.13
e = e*0.20
f = f*0.25
g = (a+b+c+d+e+f)
print 'Student name: %s' % name
print 'Course grade: %s' % g

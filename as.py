import math
choice = 'Y'
while choice == 'Y':
    choice = raw_input('Would you like to solve a quadratic equation?\nType in Y for yes\n')
    if choice != 'Y':
        break
    else:
        a = float( input( "Enter the a value for a " ) )
        b = float( input( "Enter the a value for b " ) )
        c = float( input( "Enter the a value for c " ) )
        x = (b*b) - (4*a*c)
        if x > 0:
            j = math.sqrt(x)
            r1 = (-b + j)/(2 * a)
            r2 = (-b - j)/(2 * a)
            print "The roots of this equation are:"
            print "%10.5f %10.5f" % (r1, r2)
        else:
            j = math.sqrt(-x)
            r1 = (-b/2)
            print "The roots of this equation are:"
            print "%10.5f" %(r1), '+', "%10.5f" %(j/2),'i'
            print "%10.5f" %(r1), '-', "%10.5f" %(j/2),'i'

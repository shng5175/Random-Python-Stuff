a = float( input( "Enter the a element for A: " ) )
b = float( input( "Enter the b element for A: " ) )
c = float( input( "Enter the c element for A: " ) )
d = float( input( "Enter the d element for A: " ) )
print "Matrix A"
print "|%10.5f %10.5f|" % (a, b)
print "|%10.5f %10.5f|" % (c, d)

detA = (a*d) - (c*b)
if detA == 0:
    print "This matricx is not invertible"

else:
    e = d/(detA)
    f = -b/(detA)
    g = -c/(detA)
    h = a/(detA)
    print "Matrix A'"
    print "|%10.5f %10.5f|" % (e, f)
    print "|%10.5f %10.5f|" % (g, h)

    aI = (a*e) + (b*g)
    bI = (a*f) + (b*h)
    cI = (c*e) + (d*g)
    dI = (c*f) + (d*h)

    print "Matrix A * Matrix A'"
    print "|%10.5f %10.5f|" % (aI, bI)
    print "|%10.5f %10.5f|" % (cI, dI)


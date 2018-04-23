def excessBits(b, x):
    i = 2**(b-1)
    j = i + x 
    x = bin(j)
    theList = list(x[2:])
    while (len(theList) < b):
        theList.insert(0, '0')
    return theList


def excessValue(y):
    a = len(y)
    c = ''.join(map(str, y))
    d = int(c, 2)
    e = 2**(a-1)
    f = d - e
    return f

b = input('Enter the number of bits to use: ')
x = input('Enter an integer to encode: ')
print 'You entered ', b, 'and', x
y = excessBits(b, x)
print 'The excess representation is: ', y
print 'The double-check value is: ', excessValue(y)

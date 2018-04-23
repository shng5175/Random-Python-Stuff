import time
start_time = time.time()
a = 2
b = 1
n = 1997
for i in range(2, n):
    c = a + b
    a = b
    b = c
x = int(str(c)[:5])
print x
elapsed_time = time.time() - start_time
print elapsed_time

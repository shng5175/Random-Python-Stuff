a = [8, 10, 2, 5, 20, 1]
count = 0
for i in range(len(a)):
    for j in range(i + 1, len(a)):
        if (a[i] > a[j]):
            count += 1
print count

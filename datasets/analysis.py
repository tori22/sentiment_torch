import sys
dict1 = dict()
for line in sys.stdin:
    line = line.strip()
    member = line.split(',')

    if member[2] not in dict1:
        dict1[member[2]] = 1
    else:
        dict1[member[2]] += 1
print(dict1)


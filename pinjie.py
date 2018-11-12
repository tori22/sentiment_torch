import sys
for line in sys.stdin:
    line = line.strip()
    line = line.split(' ')
    for i in line:
        print(i)

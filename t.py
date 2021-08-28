import sys

n = int(sys.stdin.readline())

info = []
people = 0

for i in range(n):
    x = list(map(int, sys.stdin.readline().split()))
    info.append(x)
    people +=x[1]

info.sort()

mid = people//2

if people%2 == 1:
    mid+=1

cnt = 0

for l, p in info:
    cnt += p

    if cnt >=mid:
        print(l)
#!/usr/bin/python
import sys
import numpy as np


def main():
    if len(sys.argv) != 3:
        print "Wrong arguments number"
        sys.exit(1)
    try:
        n = int(sys.argv[1])
        m = int(sys.argv[2])
    except ValueError:
        print "arguments should be integer values"
        sys.exit(1)
    f1 = open('build/input.txt', 'w')
    f1.write(("%d %d \n" % (n, m)))
    #a = 10 * np.random.randn(n, n)
    a = np.ones((n, n))
    for i in range(0, n):
        for j in range(0, n):
            f1.write("%.2f " % a[i, j])
        f1.write("\n")
    a = np.ones((m, m), np.float32) / (m * m)
    for i in range(0, m):
        for j in range(0, m):
            f1.write("%.2f " % a[i, j])
        f1.write("\n")
    f1.close()
    sys.exit(1)

if __name__ == '__main__':
    main()

#!/usr/bin/python
import sys
import numpy as np
import math


def main():
    if len(sys.argv) != 4:
        print "Wrong arguments number"
        sys.stdout.flush()
        sys.exit(1)
    try:
        m = int(sys.argv[1])
        n = int(sys.argv[2])
        k = int(sys.argv[3])
    except ValueError:
        print "arguments should be integer values"
        sys.exit(1)
    f1 = open('build/input.txt', 'w')
    f1.write(("%d %d %d\n" % (m, n, k)))
    a = 10 * np.random.randn(m, n)
    for i in range(0, m):
        for j in range(0, n):
            f1.write("%.2f " % a[i, j])
        f1.write("\n")
    a = 10 * np.random.randn(n, k)
    for i in range(0, n):
        for j in range(0, k):
            f1.write("%.2f " % a[i, j])
        f1.write("\n")
    f1.close()
    sys.exit(1)

if __name__ == '__main__':
    main()

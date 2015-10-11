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
    for t in range(0, 2):
        x = max(m, n) / 1024
        if m > n:
            for s in range(0, x):
                a = 10 * np.random.randn(1024, n)
                for i in range(0, 1024):
                    for j in range(0, n):
                        f1.write("%.2f " % a[i, j])
                    f1.write("\n")

            x = max(m, n) % 1024
            a = 10 * np.random.randn(x, n)
            for i in range(0, x):
                    for j in range(0, n):
                        f1.write("%.2f " % a[i, j])
                    f1.write("\n")
        else:
            for s in range(0, m):
                for l in range(0, x):
                    a = 10 * np.random.randn(1024, 1)
                    for i in range(0, 1024):
                        f1.write("%.2f " % a[i, 0])
                f1.write("\n")
        m = n
        n = k

    f1.close()
    sys.exit(1)

if __name__ == '__main__':
    main()

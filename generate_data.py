#!/usr/bin/python
import sys
import numpy as np


def main():
    if len(sys.argv) != 2:
        print "Wrong arguments number"
        sys.exit(1)
    try:
        n = int(sys.argv[1])
    except ValueError:
        print "arguments should be integer values"
        sys.exit(1)
    f1 = open('build/input.txt', 'w')
    f1.write(("%d \n" % (n)))
    #a = 10 * np.random.randn(n, n)
    a = np.ones((n))
    for i in range(0, n):
        f1.write("%.2f " % a[i])
    f1.write("\n")
    f1.close()
    sys.exit(1)

if __name__ == '__main__':
    main()

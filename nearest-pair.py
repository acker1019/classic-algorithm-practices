# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import random as rd
import math
import sys


'''
Find the nearest pair of point in a point set.
'''


def plotPoints(points, opts):
    plt.figure()
    plt.axis([0, 1, 0, 1])
    plt.plot(*np.transpose(points), opts)


def randPoints(n):
    '''
    Generate n points both in range of x, y which are [0, 1]
    '''
    points = []
    for i in range(n):
        points.append((rd.random(), rd.random()))
    return points


def nearest_pair(points):
    '''
    The input points should be sorted by x axis in ascending order.
    '''
    if len(points) <= 1:
        return sys.maxsize
    elif len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    
    mid = int(len(points) / 2)
    L = points[:mid]
    R = points[mid:]
    
    min_dist_L = nearest_pair(L)
    min_dist_R = nearest_pair(R)
    
    d = min([min_dist_L, min_dist_R])
    
    mid_point = points[mid]
    
    bar_range = (mid_point[0]-d, mid_point[0]+d)
    
    bar = []
    for i in range(mid, -1, -1):
        if points[i][0] < mid_point[0]-d:
            break
        bar.append((*points[i], 'L'))

    for i in range(mid+1, len(points), 1):
        if points[i][0] > mid_point[0]+d:
            break
        bar.append((*points[i], 'R'))
    bar.sort(key=lambda p: p[1])    
    
    for i, point_i in enumerate(bar):
        xi, yi, side_i = point_i
        if side_i == 'L':
            for j in range(i-1, -1, -1):
                xj, yj, side_j = bar[j]
                distx_ij = xi - xj
                if distx_ij > d:
                    break
                elif side_j == 'R':
                    dist_ij = math.sqrt(math.pow(xi-xj, 2) + math.pow(yi-yj, 2))
                    d = min([d, dist_ij])
            for j in range(i+1, len(bar), 1):
                xj, yj, side_j = bar[j]
                distx_ij = xj - xi
                if distx_ij > d:
                    break
                elif side_j == 'R':
                    dist_ij = math.sqrt(math.pow(xj-xi, 2) + math.pow(yj-yi, 2))
                    d = min([d, dist_ij])
    
    return d                



# Generate a set of points randomly
points = randPoints(21)

# Draw the input set of points
plotPoints(points, 'bo')
plt.show()

min_dist = nearest_pair(points)
print('minimum distance: ', min_dist)


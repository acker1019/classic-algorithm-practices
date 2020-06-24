# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import random as rd
import math


'''
For a set of points, find the convex hull.
A convex hull means that a minimized convex polygon includes all the points.
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


def findConvexHull(points):
    '''
    The Graham's scan
    '''
    
    points = points[:]
    
    '''
    1. Find the axle center
    
    Find the most left point. If there are points with same x value,
    choose the most bottom one.
    
    The reason is that the first point in counterclockwise has the smallest tangent value
    and the last point in counterclockwise has the highest tangent value.
    '''
    xs = [x for x, y in points]
    ys = [y for x, y in points]
    min_x = min(xs)
    candi = [i for i, x in enumerate(xs) if x == min_x]
    candi_y = [ys[i] for i in candi]
    axis_i = ys.index(min(candi_y))
    axis = (xs[axis_i], ys[axis_i])
    del points[axis_i]
    
    # Draw to check
    plotPoints(points, 'bo')
    plt.plot(*axis, 'o', c='orange')
    plt.show()
    
    
    '''
    2. Sort points counterclockwisely with the axis as the center.
    '''
    # Figure out the tangent value of each point when the center is the axis decided above.
    tans = [(y - axis[1]) /(x - axis[0]) for x, y in points]
    print(tans)
    
    # Sort points counterclockwisely according to their tangent value.
    s = [(p, t) for p, t in zip(points, tans)]
    s.sort(key=lambda tpl: tpl[1])
    points = [p for p, _ in s]
    print(points)
    
    # Draw to check
    for point in points:
        plotPoints(points, 'bo')
        plt.plot(*axis, 'o', c='orange')
        plt.plot([axis[0], point[0]], [axis[1], point[1]])
        plt.show()
    
    
    '''
    3. Find the convex hull
    '''
    # Take the first 3 points as initialization
    convexHull = [axis]
    
    # Scan the points counterclockwisely
    for next_point in points:
        while len(convexHull) >= 2 and isRight(convexHull[-2], convexHull[-1], next_point):
            '''
            If the new point is on the right side of vector (convexHull[-2], convexHull[-1]),
            it means that the angle between the 3 points is bigger than 180°.
            In the case of convex hull,
            the angle between any 3 neighbors is always smaller or equivalent to 180°.
            Once the angle over 180° is detected,
            the last point will be deleted to keep the shape a convex hull.
            '''
            convexHull.pop()
        
        convexHull.append(next_point)
        
        # plot this step for observation
        plotPoints(points, 'bo')
        plt.plot(*axis, 'o', c='orange')
        plt.plot(*list(zip(*convexHull)))
        plt.show()
    
    # Make the path a closed shape and return the result.
    convexHull.append(convexHull[0])
    return convexHull
    
    
def isRight(p1, p2, p3):
    '''
    For a vector V(p2.x - p1.x, p2.y - p1.y), the following statements hold:
        1. If the outcome > 0, the p3 is on the right side of vector V.
        2. If the outcome < 0, the p3 is on the left side of vector V.
        3. If the outcome = 0, the p3 is right on the extension of vector V.
    
    For details:
    https://math.stackexchange.com/questions/757591/how-to-determine-the-side-on-which-a-point-lies
    '''
    outcome = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    return outcome > 0



# Generate a set of points randomly
points = randPoints(12)

# Draw the input set of points
plotPoints(points, 'bo')
plt.show()

# find the convex hull
convexHull = findConvexHull(points)

# Draw the result
plotPoints(points, 'ro')
plt.plot(*list(zip(*convexHull)))
plt.show()




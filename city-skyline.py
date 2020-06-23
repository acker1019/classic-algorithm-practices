# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import random as rd
from bisect import bisect_left, bisect_right
import numpy as np
import json


def randCity(n, building_area_limits=(0.02, 0.07), building_height_limits=(0.04, 0.6)):
    '''
    purpose:
    Randomly generate n buildings for a city in range of [0.0, 1.0].
    The buildings are sorted with their left bound.
    
    details:
    Each of the generated building has its area as the building_area_limits gives.
    The building_height_limits constraints its height range.
    The width of the building can be derived by building_area_limits/building_height_limits.
    If the width is bigger than 1.0, it will be fixed to 1.0.
    If the the right bound of a building is out of 1.0,
    the whole building will be shifted to left to match the right bound of the city.
    '''
    city = []
    for i in range(n):
        area = rd.uniform(*building_area_limits)
        h = rd.uniform(*building_height_limits)
        x1 = rd.random()
        x2 = x1 + min([1, area / h])
        fix = max([0, x2 - 1])
        city.append([(x1 - fix, 0), (x1 - fix, h), (x2 - fix, h), (x2 - fix, 0)])
        
    city.sort(key=lambda building: building[0][0])
        
    return city


def draw_city(city, max_building_height_limit=0.6):
    '''
    Draw out the city for observation.
    '''
    plt.figure()
    # color switcher
    cmap = plt.get_cmap('Paired')
    N = len(city)
    
    # plot ground
    plt.plot([0, 1], [0, 0], c='black')
    
    # plot buildings
    axis = plt.gca()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, max_building_height_limit)
    for i, shape in enumerate(city):
        # chose color
        color = cmap(float(i)/N)
        
        # plot line
        xs, ys = np.transpose(shape)
        plt.plot(xs, ys, c=color)
    


def merge(contour, building):
    '''
    Merge a building into the contour.
    
    1. Find the parts of city overlaping the building.
    2. Generate points for intersection of building edges.
    3. Keep the parts heigher than the building and delete the lower one but 
    4. Keep the first point as the head point for those points
        whose heights equal to the building's height.
    '''
    # Extract the parameters
    contour_x, contour_y = contour
    building_x0, building_x1, building_y = building
    
    # Search the location of the building related to the city.
    left_bound = bisect_left(contour_x, building_x0)
    right_bound = bisect_right(contour_x, building_x1)
    
    # From left bound of the building to its right bound.
    i = left_bound
    # A flag to determine to keep the head point or delete the following points.
    hasHead = False
    
    if building_y > contour_y[i]:
        '''
        If the left-top corner of the building is heigher than the contour,
        keep the left bound.
        The corner will also be the head point.
        '''
        
        hasHead = True
        
        # add intersection point
        contour_x.insert(i, building_x0)
        contour_y.insert(i, contour_y[i])
        i += 1
        right_bound += 1
        
        # add building point
        contour_x.insert(i, building_x0)
        contour_y.insert(i, building_y)
        i += 1
        right_bound += 1
    
    while i < right_bound:
        '''
        Determine the overlaping parts.
        '''
        if contour_y[i] < building_y:
            '''
            If the parts of city lower than the building.
            '''
            if hasHead:
                del contour_x[i]
                del contour_y[i]
                right_bound -= 1
            else:
                hasHead = True
                contour_y[i] = building_y
                i += 1
        else:
            '''
            If the parts of city heigher than the building.
            '''
            i += 1
    
    if i < len(contour_x):
        '''
        If the right bound of the building is in the range of the city.
        '''
        if building_y > contour_y[i]:
            '''
            If the right-top corner of the building is heigher than the contour,
            keep the right bound.
            '''
            
            # add building point
            contour_x.insert(i, building_x1)
            contour_y.insert(i, building_y)
            i += 1
            
            # add intersection point
            contour_x.insert(i, building_x1)
            contour_y.insert(i, contour_y[i])
            i += 1
    else:
        '''
        If the right bound of the building is out of the range of the city,
        simplely append the right bound to the contour.
        '''
        contour_x.append(building_x1)
        contour_y.append(building_y)
        contour_x.append(building_x1)
        contour_y.append(0)


def findSkyline(city):
    '''
    Find the city skyline constituded by several buildings.
    '''
    
    # Initialize the city skyline by the ground (the horizon) .
    contour = [[0, 1], [0, 0]]
    
    for building in city:
        '''
        Append buildings to the contour one by one.
        '''
        
        # Extract the parameters.
        building_x0 = building[0][0]
        building_x1 = building[-1][0]
        building_y = building[1][1]
        
        # Draw the contour before building appending
        draw_city([[(x, y) for x, y in zip(*contour)]])
        plt.plot(*np.transpose(building), c='orange')
        plt.show()
        
        # Merge the building to the contour.
        merge(contour, (building_x0, building_x1, building_y))
        
        # Draw the contour after building appending
        draw_city([[(x, y) for x, y in zip(*contour)]])
        plt.show()
    
    # Transpose
    contour = [[(x, y) for x, y in zip(*contour)]]
    return contour


def readCity(fpath):
    '''
    Read a existed city.
    '''
    with open(fpath, 'r') as jfile:
        arr = json.load(jfile)
    return arr




city = randCity(n=9)
#city = readCity('city.json')

print('city:\n', city)

draw_city(city)
plt.show()

skyline = findSkyline(city)

print('\nskyline:\n', skyline)

draw_city(skyline)
plt.show()


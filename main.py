# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:01:55 2022

@author: Alkios
"""

import os
os.chdir('C:/Users/Alkios/Downloads/signals')


from scipy.spatial import Voronoi
from matplotlib.patches import Polygon

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('lena.png')
plt.imshow(img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray,250,250)
plt.imshow(edges,cmap = 'gray')

dst = cv.GaussianBlur(edges,(25,25), 10)
plt.imshow(dst,cmap = 'gray')

ctest = cv.subtract(dst,edges)
plt.imshow(ctest,cmap = 'gray')


modified_test = ctest/np.sum(ctest)
np.sum(modified_test)

# Create a flat copy of the array
flat = modified_test.flatten()

# Then, sample an index from the 1D array with the
# probability distribution from the original array
sample_index = np.random.choice(a=flat.size, p=flat)


img_modif = cv.flip(img, 0)


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

    
def voronoi_polygons(n=256):
    random_seeds = points_coord
    vor = Voronoi(random_seeds)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    return polygons



def plot_color(t, as_str=True, alpha=0.5):
    rgb = new_l3[t]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        # Normalize & listify
        return list(np.array(rgb)/255) + [alpha]
    


def plot_polygons2(polygons, ax=None, alpha=0.5, linewidth=0.7, saveas=None, show=True):
    # Configure plot 
    if ax is None:
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis("equal")

    # Set limits
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(0,img.shape[0])

    # Add polygons 
    m = 0
    for poly in polygons:
        colored_cell = Polygon(poly,
                               linewidth=linewidth, 
                               alpha=alpha,
                               facecolor=plot_color(m, as_str=False, alpha=1),
                               edgecolor="black")
        ax.add_patch(colored_cell)
        m += 1

    if not saveas is None:
        plt.savefig(saveas)
    if show:
        plt.show()
    return ax 



nbpoints = [500, 1000, 1500, 2000, 2500, 5000, 15000]

for i in nbpoints :

    sample_index = np.random.choice(a=flat.size, size = i, p=flat, replace=False)
    adjusted_index = np.unravel_index(sample_index, modified_test.shape)
    
    points_coord = np.transpose(np.array([adjusted_index[1],img.shape[0] -1 - adjusted_index[0]]))
    
    new_l2 = []
    
    for i in points_coord:
        new_l2.append(list(img_modif[i[1],i[0]]))
    
    new_l3 = []
    
    for l in range(len(new_l2)):
        new_l3.append([new_l2[l][2],new_l2[l][1],new_l2[l][0]])
    
    new_l3 = np.array(new_l3)
    
    
    test = plot_polygons2(voronoi_polygons(i),
                              alpha = 1,
                              show = False)

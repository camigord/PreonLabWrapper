import numpy as np

'''
Collision detection using SAT (separating axis theorem)
'''

def collide(p1, p2):
    '''
    Return True if collision is detected

    p1 and p2 are lists of ordered pairs, the vertices of the polygons in the
    counterclockwise direction.
    '''

    p1 = [np.array(v, 'float64') for v in p1]
    p2 = [np.array(v, 'float64') for v in p2]

    edges = edges_of(p1)
    edges += edges_of(p2)
    orthogonals = [orthogonal(e) for e in edges]

    push_vectors = []
    for o in orthogonals:
        separates = is_separating_axis(o, p1, p2)

        if separates:
            return False
        else:
            continue

    return True

def is_separating_axis(o, p1, p2):
    """
    Return True and the push vector if o is a separating axis of p1 and p2.
    Otherwise, return False and None.
    """
    min1, max1 = float('+inf'), float('-inf')
    min2, max2 = float('+inf'), float('-inf')

    for v in p1:
        projection = np.dot(v, o)

        min1 = min(min1, projection)
        max1 = max(max1, projection)

    for v in p2:
        projection = np.dot(v, o)

        min2 = min(min2, projection)
        max2 = max(max2, projection)

    if max1 >= min2 and max2 >= min1:
        return False
    else:
        return True

def edges_of(vertices):
    """
    Return the vectors for the edges of the polygon p.

    p is a polygon.
    """
    edges = []
    N = len(vertices)

    for i in range(N):
        edge = vertices[(i + 1)%N] - vertices[i]
        edges.append(edge)

    return edges

def orthogonal(v):
    """
    Return a 90 degree clockwise rotation of the vector v.
    """
    return np.array([-v[1], v[0]])

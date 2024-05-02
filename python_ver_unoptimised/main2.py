import random
import matplotlib.pyplot as plt
import math
from functions import *




from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Triangulation_2 import VERTEX
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_circulator
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_handle
from CGAL.CGAL_Kernel import Ref_int
from CGAL.CGAL_Triangulation_2 import Ref_Locate_type_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2



n=500

# Generate a list of 100 random points in the square (0,0) to (1,1)
points = [(random.random(), random.random()) for i in range(n)]
	

# Perform the Delaunay triangulation
T = Delaunay_triangulation_2()
points = [Point_2(*p) for p in points]
T.insert(points)

# Extract the edges of the triangulation
edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]

draw_connected_edges(T,points[1],edges)



#Move the point and plot the triangulation
#move_point_and_recalculate(T,points[1],0.1)
add_point_plot(T,points[1],points,0.1)

'''
# Plot the triangulation
fig, ax = plt.subplots()
for edge in edges:
    ax.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 1,color = 'black')
ax.scatter([p.x() for p in points], [p.y() for p in points], s=10)

plt.show()
'''

plt.show()








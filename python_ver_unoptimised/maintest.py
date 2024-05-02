import random
import matplotlib.pyplot as plt
import math
from functions3 import *
from drops import *



from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Triangulation_2 import VERTEX
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_circulator
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_handle
from CGAL.CGAL_Kernel import Ref_int
from CGAL.CGAL_Triangulation_2 import Ref_Locate_type_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2



n= 10
t0=1
r=0.00001
# Generate a list of 100 random points in the square (0,0) to (1,1)
points = [(random.random(), random.random()) for i in range(n)]
	

# Perform the Delaunay triangulation
T = Delaunay_triangulation_2()
points = [Point_2(*p) for p in points]
T.insert(points)


# Extract the edges of the triangulation
#edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]



'''
Testing the add and remove point plots
'''
'''

# Point to remove
x = points[0].x()
y = points[0].y()




remove_plot(T,x,y,points)


#Point to add
add_plot(T,0.5,0.5,points)
'''




#simple_plot(T,points)
#blockPrint()
#draw_connected_edges(T,points[0],edges)
df= create_dataframe(T,t0,r)
#bubble_plot4(df)
#next_tc = all_min_tcoal_with_neighbours(df,n)


new_df = growth_dataframe_mod(df,n)
print('\n')
print('DATAFRAME AFTER GROWTH')
print(new_df)










bubble_plot4(new_df)
newer_df = coal_event_df(T,new_df,n)
print('DATAFRAME AFTER COAL')
print(newer_df)
bubble_plot4(newer_df)

print('DATAFRAME AFTER SECOND COAL')
#second_coal_df = growth_dataframe_mod(newer_df,n)
second_coal_df = coal_event_df(T,newer_df,n)
print(second_coal_df)
bubble_plot4(second_coal_df)


print('CoalTime list ---------------------',tn)






#draw_connected_edges(T,points[1],edges)


#Move the point and plot the triangulation
#move_point_and_recalculate(T,points[1],0.1)
#add_point_plot(T,points[1],points,0.1)

'''
# Plot the triangulation
for edge in edges:
    plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 1,color = 'black')
plt.scatter([p.x() for p in points], [p.y() for p in points], s=10)

'''






plt.show()











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
'''
NEWEST VERSION OF MAIN
'''


n=15	
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

# Removing the point
remove_plot(T,x,y,points)

#Adding the point
add_plot(T,0.5,0.5,points)
'''




df= create_dataframe(T,t0,r)


new_df = growth_dataframe_mod(df,n)
print('DATAFRAME AFTER GROWTH')
print(new_df)
print('\n')
#bubble_plot4(new_df)





first_coal = coal_event_df(T,df,n)
print('DATAFRAME AFTER COAL')
print(first_coal)
bubble_plot4(first_coal)


second_coal_before = growth_dataframe_mod(first_coal,n-1)
print(second_coal_before)
#bubble_plot4(second_coal_before)



print('DATAFRAME AFTER SECOND COAL')
#second_coal_df = growth_dataframe_mod(first_coal,n-1)
second_coal_df2 = coal_event_df(T,first_coal,n-1)
bubble_plot4(second_coal_df2)





print('DATAFRAME AFTER SECONDGROWTH')
third_coal_before = growth_dataframe_mod(second_coal_df2,n-2)
print(third_coal_before)
#bubble_plot4(third_coal_before)





print('DATAFRAME AFTER THIRDCOAL')
third_coal_df2 = coal_event_df(T,second_coal_df2,n-2)
print(third_coal_df2)
bubble_plot4(third_coal_df2)



print('DATAFRAME AFTER THIRDGROWTH')
fourth_coal_before = growth_dataframe_mod(third_coal_df2,n-3)
print(fourth_coal_before)
#bubble_plot4(fourth_coal_before)





print('DATAFRAME AFTER FOURTHCOAL')
fourth_coal_df2 = coal_event_df(T,third_coal_df2,n-3)
print(third_coal_df2)
bubble_plot4(fourth_coal_df2)





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











import random
import matplotlib.pyplot as plt
import math
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Triangulation_2 import VERTEX
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_circulator
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_handle
from CGAL.CGAL_Kernel import Ref_int
from CGAL.CGAL_Triangulation_2 import Ref_Locate_type_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2



def diff(old,new):
    return list(set(new) - set(old))




def move_point_and_recalculate(T, point,points,distance):
    # Generate a random angle in radians
    angle = random.uniform(0, 2 * math.pi)
    
    # Calculate the new x and y coordinates of the point
    x = point.x() + distance * math.cos(angle)
    y = point.y() + distance * math.sin(angle)
    
    # Creating a point and getting its vertex
    new_point = Point_2(x,y)
    old_vertex = T.nearest_vertex(point)
    
    
    # Remove the point from the triangulation
    T.remove(old_vertex)
    points.remove(point)
    
    
    # Insert the point with the new coordinates into the triangulation
    T.insert(iter([new_point]))
    points.append(new_point)
   
    # Return the new Point_2	
    return T.nearest_vertex(new_point)


def add_point_plot(T,point,points,distance) : 
	# Get the set of edges before moving the point
	edges_before = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
	
	
	# Plot the edges of the edges before change in black
	for edge in edges_before:
    		plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 0.5,color = 'black')
    		
    	
    	
	
	# Move the point and recalculate the triangulation
	moved_point = move_point_and_recalculate(T, point,points, distance)
	
	# Get the set of edges after moving the point
	edges_after = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
	# Scatter plot the vertices
	plt.scatter([p.x() for p in points], [p.y() for p in points], s=10, color = 'black')



	# Find the edges that have changed
	# Create a list of Point coordinates instead of Point_2 objects
	list_of_coordinates_before = []
	for i in edges_before:
		for j in i :
			list_of_coordinates_before.append((j.x(),j.y())) 
	list_of_coordinates_before = [(list_of_coordinates_before[i],list_of_coordinates_before[i+1]) for i in range(0,len(list_of_coordinates_before),2)]
	
	
	list_of_coordinates_after = []
	for i in edges_after:
		for j in i :
			list_of_coordinates_after.append((j.x(),j.y())) 
	list_of_coordinates_after = [(list_of_coordinates_after[i],list_of_coordinates_after[i+1]) for i in range(0,len(list_of_coordinates_after),2)]
	
	
	#print('List of coordinates before', list_of_coordinates_before)
	#print('\n' + ' List of coordinates after', list_of_coordinates_after)
	
	
	# Find the changed coordinates
	
	
	changed_edges = diff(list_of_coordinates_before,list_of_coordinates_after)
	#print('\n' + ' Coordinates that have changed are ',changed_edges )
	#print(' The number of edges that have been changed are ', len(changed_edges))
	changed_edges_list = [(elem1, elem2) for elem1, elem2 in changed_edges]
	#print('\n',changed_edges_list)
	
	not_changed_edges = diff(list_of_coordinates_after,list_of_coordinates_before)
	not_changed_edges_list = [(elem1, elem2) for elem1, elem2 in not_changed_edges]
	

	
	
	# Plot the edges that have changed in a different color
	for edge_co in changed_edges:
	
		if edge_co in changed_edges:
			plt.plot(*zip(*edge_co),color = 'red',lw = 0.5)
			#plt.scatter(*zip(*edge_co), s=10,color = 'black') 
	
	
	

	


	# Color the edges of the removed point
	
	old_vertex_handle = T.nearest_vertex(point)
	
	
	
	
def draw_connected_edges(T,point,edges):
	vertex_handle = T.nearest_vertex(point)
	# Get the list of coordinates of all vertex
	list_of_coordinates = []
	for i in edges:
		for j in i :
			list_of_coordinates.append((j.x(),j.y())) 
	list_of_coordinates= [(list_of_coordinates[i],list_of_coordinates[i+1]) for i in range(0,len(list_of_coordinates),2)]
	
	# Find the ones which begin or end at our point
	px = point.x()
	py= point.y()
	list_coordinates_to_plot = []
	for i in list_of_coordinates:
		print('i-------------------------',i)
		for j in i:
			if px == j[0] and py == j[1]:
				list_coordinates_to_plot.append(i)
	print(list_coordinates_to_plot)
	
	for old_edge in list_coordinates_to_plot:
	
		if old_edge in list_coordinates_to_plot:
			plt.plot(*zip(*old_edge),color = 'green',lw = 1.0)
			
			

			
			
	
	
	
	
	
	

	














'''

  
def add_point_plot(T,point,points,distance) : 
	# Get the set of edges before moving the point
	edges_before = set(T.finite_edges())

	# Move the point and recalculate the triangulation
	moved_point = move_point_and_recalculate(T, point,points, distance)
	edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
	
	# Get the set of edges after moving the point
	edges_after = set(T.finite_edges())

	# Find the edges that have changed
	changed_edges = edges_after.symmetric_difference(edges_before) # changed edges is a dictionnary of tuples
	changed_edges_list = list(changed_edges)
	
	
	
	
	
	# Plot the edges that have changed in a different color
	for edge in edges:
		print('EDGE ---------------',edge)
		if edge in changed_edges_list:
			plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], color = 'red',lw=10)
			#print('EDGE ---------------',edge)	
		else:
			plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], color ='black',lw = 0.5)
			plt.scatter([p.x() for p in points], [p.y() for p in points], s=10)  
		
	
'''	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

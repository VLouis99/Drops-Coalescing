import random
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import vaex
import plotly
import plotly.graph_objects as go
from matplotlib.collections import EllipseCollection
from global_variables import *

from drops import *
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Triangulation_2 import VERTEX
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_circulator
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_handle
from CGAL.CGAL_Kernel import Ref_int
from CGAL.CGAL_Triangulation_2 import Ref_Locate_type_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
import time

tn = [1]

'''
Notes :
- List append is constant time while numpy append is linear
- 100 points 5 seconds, 500 points 25 seconds ,1000 points 60 seconds
			
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def diff(old,new):
    return list(set(new) - set(old))
   



def edge_to_point(l,point):
	new_list = []
	for i in l:
			new_list.append(i[0])
			new_list.append(i[1])
	# Remove all the instances of the concerned point in the list
	px=point.x()
	py= point.y()
	new_list = [*set(new_list)]
	new_list.remove((px,py))
	
	return new_list
		
    


def listofarray_to_listofscalar(l):
	return [i[0] for i in l]



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Plotting functions and delaunay triangulation manipulation
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#			
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def simple_plot(T,points):
    edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
    for edge in edges:
        plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 0.5,color = 'black')
        plt.scatter([p.x() for p in points], [p.y() for p in points], s=10)





def bubble_plot(df):
	fig,ax = plt.subplots()
	for r in df.Radius.values:
		rr=r
		# radius in data coordinates:
		# radius in display coordinates:
		r_ = ax.transData.transform([rr,0])[0] - ax.transData.transform([0,0])[0]
		# marker size as the area of a circle
		marker_size = np.pi * (r_/10)**2
		print('**********msize***************',marker_size)
		fig = go.Figure(data=[go.Scatter(x=df.x.values, y=df.y.values,mode='markers',marker_size=marker_size)])
	fig.show()


def bubble_plot2(df):
	fig,ax = plt.subplots(figsize=[7,7])

	# important part:
	# calculate the marker size so that the markers touch
	for i in range(len(df.Radius.values)):
		rr=df.Radius.values[i]
		print(rr)
		# radius in data coordinates:
		# radius in display coordinates:
		r_ = ax.transData.transform([rr,0])[0] - ax.transData.transform([0,0])[0]
		# marker size as the area of a circle
		marker_size = np.pi * r_**2

		ax.scatter(df.x.values[i], df.y.values[i], s=marker_size,alpha=0.3, edgecolors='black')
	
	plt.show()
	
	
def bubble_plot3(df):
	f = plt.figure(figsize=(7, 7), dpi=100)
	ax = f.add_subplot(1,1,1)
	for i in range(len(df.Radius.values)):
		rr=df.Radius.values[i]
		L = 2*rr # Diameter of a circle in data units 
		# marker_radius is L/2 in points.
		marker_radius = (ax.transData.transform((0,0))
				         -ax.transData.transform((L/2,0)))[0]
		marker_area = np.pi*marker_radius**2
		ax.scatter(df.x.values, df.y.values, color='blue', edgecolors='black', s=marker_area)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()
		
# df is a tuple
def bubble_plot4(df):
	f = plt.figure(figsize=(7, 7), dpi=100)
	ax = f.add_subplot(1,1,1)
	marker_area_list = []
	#df = df[0]
	for i in range(len(df.Radius.values)):
		rr=df.Radius.values[i]
		L = 2*rr # Diameter of a circle in data units 
		# marker_radius is L/2 in points.
		marker_radius = (ax.transData.transform((0,0))
				         -ax.transData.transform((L/2,0)))[0]
		marker_area = 2.5*marker_radius**2
		marker_area_list.append(marker_area)
	for j in range(len(marker_area_list)):
		ax.scatter(df.x.values[j], df.y.values[j], color='blue', edgecolors='black', s=marker_area_list[j])
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()
	
	
	
def bubble_plot5(df):
	size = df.Radius.values
	f = plt.figure(figsize=(7, 7), dpi=100)
	ax = f.add_subplot(1,1,1)
	color ='b' #np.ones(100) * 40

	offsets = list(zip(df.x.values, df.y.values))
	ax.add_collection(EllipseCollection(widths=size*2, heights=size*2, angles=0, units='xy',
		                                   color = 'blue',edgecolor ='black',
		                                   offsets=offsets, transOffset=ax.transData))
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()
	




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_point(T,x,y,points):
	new_point = Point_2(x,y)
	T.insert(iter([new_point]))
	points.append(new_point)
	

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def remove_point(T,x,y,points):
	removed_point = Point_2(x,y)
	removed_vertex= T.nearest_vertex(removed_point)
	T.remove(removed_vertex)
	points.remove(removed_point)
	

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def Del_removal(T,x,y):
	removed_point = Point_2(x,y)
	removed_vertex= T.nearest_vertex(removed_point)
	T.remove(removed_vertex)
	
	
def Del_add(T,x,y):
	new_point = Point_2(x,y)
	T.insert(iter([new_point]))
	
	
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_plot(T,x,y,points):
    edges_before = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
    #for edge in edges_before:
    	#plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 0.5,color = 'black')
    	
    # Scatter plot the vertices
    plt.scatter([p.x() for p in points], [p.y() for p in points], s=10, color = 'black')
    		
    # Find the edges that have changed
    # Create a list of Point coordinates instead of Point_2 objects
    list_of_coordinates_before = []
    for i in edges_before:
        for j in i :
            list_of_coordinates_before.append((j.x(),j.y())) 
			
    list_of_coordinates_before = [(list_of_coordinates_before[i],list_of_coordinates_before[i+1]) for i in range(0,len(list_of_coordinates_before),2)]
	
	
    add_point(T,x,y,points)
    edges_after=[ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
    	
    list_of_coordinates_after = []
    for i in edges_after:
        for j in i :
        	list_of_coordinates_after.append((j.x(),j.y())) 
    list_of_coordinates_after = [(list_of_coordinates_after[i],list_of_coordinates_after[i+1]) for i in range(0,len(list_of_coordinates_after),2)]
	
	
    #print('List of coordinates before', list_of_coordinates_before)
    #print('\n' + ' List of coordinates after', list_of_coordinates_after)
	
	
    # Find the changed coordinates
	
	
    changed_edges = diff(list_of_coordinates_before,list_of_coordinates_after)
    #print('Changed edges -------------', changed_edges)
    #print('\n' + ' Coordinates that have changed are ',changed_edges )
    #print(' The number of edges that have been changed are ', len(changed_edges))
    changed_edges_list = [(elem1, elem2) for elem1, elem2 in changed_edges]
    #print('\n',changed_edges_list)
	

	
	
    # Plot the edges that have changed in a different color
    for edge_co in changed_edges:
    	if edge_co in changed_edges:
	    	plt.plot(*zip(*edge_co),color = 'red',lw = 0.5)
	    	
    edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]    	
    for edge in edges:
    	plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 0.1,color = 'black')
    	
    # Scatter plot the vertices
    plt.scatter([p.x() for p in points], [p.y() for p in points], s=10, color = 'black')
	
	

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def remove_plot(T,x,y,points):
    edges_before = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
    for edge in edges_before:
    	plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 0.5,color = 'black')
    	
    # Scatter plot the vertices
    plt.scatter([p.x() for p in points], [p.y() for p in points], s=10, color = 'black')
    		
    # Find the edges that have changed
    # Create a list of Point coordinates instead of Point_2 objects
    list_of_coordinates_before = []
    for i in edges_before:
        for j in i :
            list_of_coordinates_before.append((j.x(),j.y())) 
			
    list_of_coordinates_before = [(list_of_coordinates_before[i],list_of_coordinates_before[i+1]) for i in range(0,len(list_of_coordinates_before),2)]
	
	
    remove_point(T,x,y,points)
    edges_after=[ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
    	
    list_of_coordinates_after = []
    for i in edges_after:
        for j in i :
        	list_of_coordinates_after.append((j.x(),j.y())) 
    list_of_coordinates_after = [(list_of_coordinates_after[i],list_of_coordinates_after[i+1]) for i in range(0,len(list_of_coordinates_after),2)]
	
	
    #print('List of coordinates before', list_of_coordinates_before)
    #print('\n' + ' List of coordinates after', list_of_coordinates_after)
	
	
    # Find the changed coordinates
	
	
    changed_edges = diff(list_of_coordinates_after,list_of_coordinates_before) # ORDER CHANGED COMPARED TO ADD_PLOT
    #print('Changed edges -------------', changed_edges)
    
    changed_edges_list = [(elem1, elem2) for elem1, elem2 in changed_edges]
 
	

	
	
    # Plot the edges that have changed in a different color
    for edge_co in changed_edges:
    	if edge_co in changed_edges:
	    	plt.plot(*zip(*edge_co),color = 'green',lw = 0.5)
        	
    #Redraw the new triangulation after point is removed
    edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
    for edge in edges:
    	plt.plot([edge[0].x(), edge[1].x()], [edge[0].y(), edge[1].y()], lw = 1,color = 'black',linestyle ='dotted')




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#  
	
  
	


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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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
	
	
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#	
	
def draw_connected_edges(T,point,edges):
	list_coordinates_to_plot = get_neighbours(T,point,edges)
	
	for edge in list_coordinates_to_plot:
	
		if edge in list_coordinates_to_plot:
			plt.plot(*zip(*edge),color = 'blue',lw = 3)
	
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_neighbours(T,point,edges):
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
	neighbour_coordinates = []
		
	for i in list_of_coordinates:
		for j in i:
			if px == j[0] and py == j[1]:
				neighbour_coordinates.append(i)
	return neighbour_coordinates # under edge form
	
	
		



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Calculating the drops
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#			
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#			

			
def distance(edge):
	return np.sqrt((edge[0].x() - edge[1].x())**2  + (edge[0].y() - edge[1].y()) **2)
	

def distance_coor(coordinate1,coordiante2):
	return np.sqrt((coordinate1[0] - coordiante2[0])**2  + (coordinate1[1] - coordiante2[1]) **2)
	
	
	
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_dataframe(T,t,r):
	all_drops = define_all_drops(T,t,r)
	indices = []
	radius = []
	x = []
	y = []
	xy=[]
	neighbours = []
	edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
	# Create a list of each element of a drop
	for drop in all_drops:
		indices.append(drop.index)
		radius.append(drop.radius)
		x.append((drop.point.x()))
		y.append((drop.point.y()))
		xy.append((drop.point.x(),drop.point.y()))
		neighbours.append(data_frame_neighbours(T,drop.point,edges))
		
	
				
	indices = np.array(indices)
	radius = np.array(radius)
	x = np.array(x)
	y = np.array(y)
	xy = np.array(xy)
	neighbours = np.array(neighbours)
	df = vaex.from_arrays(Indicices=indices, Radius = radius, x = x, y = y, xy = xy , neighbours=neighbours)
	return df



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def growth_dataframe_mod(df,n): 
	next_tc = all_min_tcoal_with_neighbours(df,n)	
	print('next_tc',next_tc)
	radiusus = df.Radius.values
	Radiusus = []
	for i in range(n):
		rr = radius_calculator(radiusus[i],next_tc)  # *tn[-1]
		Radiusus.append(rr)
	df['Radius'] = np.array(Radiusus)
	#print('Radiusus',df.Radius.values)
	if not isinstance(df.Radius.values[0], np.floating):
		df['Radius'] = np.array(listofarray_to_listofscalar(Radiusus))
	return df	

def growth_dataframe_opt(df,n):
	next_tc = df.tcoal.values
	next_tc = np.array(next_tc).min()
	if next_tc <= 1e-250:
		return df
	print('*******************************WENT TO NEXT STEP********************************')
	next_tc = next_tc + tn[-2] #CHECK Tn
	print('NEXT TC IS', next_tc)
	radiusus = df.Radius.values
	Radiusus = []
	for i in range(n):
		rr = radius_calculator(radiusus[i],next_tc)  
		Radiusus.append(rr)
	df['Radius'] = np.array(Radiusus)
	if not isinstance(df.Radius.values[0], np.floating):
		df['Radius'] = np.array(listofarray_to_listofscalar(Radiusus))
	return df	
	

	
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#	

def radius_calculator(r,t):
	return (t/tn[-2])**(1/3) * r
	
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_pos_in_df(df,coord):
	for pos in range(len(df)):
		if df.xy.values[pos][0] == coord[0]:
			return pos

def coal_event_df(T,df,n):
	
	minimum_tcoal = np.array(df.tcoal.values).min()
	if minimum_tcoal > 1e-250:	
		tn.append(minimum_tcoal)
	else :
		tn.append(tn[-1])
	i_list = []
	df_rows = df[df.tcoal.values == minimum_tcoal]
	'''
	for i in range(n):
		if df.tcoal.values[i] == minimum_tcoal:
			i_list.append(i)
	print("ILIST = ",i_list)
	'''
	
	for i in range(n):
		if df.tcoal.values[i] == minimum_tcoal:
			i_list.append(i)
	i_list.append(get_pos_in_df(df,df.Coal_Drop.values[i_list[0]]))
	#i_list.append(get_pos_in_df(df,df.xy.values[i_list[0]]))
	#print("ILIST = ",i_list)
	
	pos1 = i_list[0]
	pos2 = i_list[1]
	R1 = df[pos1][1]
	R2 = df[pos2][1]
	
	
	# Calculate the center of mass coordinates of these two drops
	c1 = df[pos1][4]
	c2 = df[pos2][4]
	cdm = center_of_mass_coor(c1, c2, R1, R2)
	new_point = Point_2(cdm[0],cdm[1])
	
	

	# Calculate the new drops radius
	R =  (R1**3 + R2**3)**(1/3)
	
	
	# Remove the two old drops from the delaunay triangulation and the dataframe
	Del_removal(T,c1[0],c1[1])
	Del_removal(T,c2[0],c2[1])
	
	# Remove one of the drops from the DF ( Careful, have to remove the second drop )
	#df = df[df.x != c1[0]]
	df = df[df.x != c2[0]]
	
	
	
	
	# Add the new drop to the delaunay triangulation
	Del_add(T,cdm[0],cdm[1])
	
	
	# Calculate the neighbours of the drops
	edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
	d_neigh = data_frame_neighbours(T,new_point,edges)
	
	
			
	
	
		
	# Change the remaning drop to the new drop that has formed
	#The where syntax is where(if, then, else) or where(condition, if condition satisfied, otherwise).
	#df['Radius'] = df.func.where(df.tcoal ==minimum_tcoal, R, 10) # Appparently does not take more time to use where several times
	#df['x'] = df.func.where(df.tcoal ==minimum_tcoal, cdm[0], df.x)
	#df['y'] = df.func.where(df.tcoal ==minimum_tcoal, cdm[1], df.y)
	#df['xy'] = df.func.where(df.tcoal==minimum_tcoal,cdm, df.xy.values)
	
	new_radiusus = []
	new_x = []
	new_y = []
	new_neighbours = []
	new_xy = []
	
	
	
	
	for i in range(len(df)):
		if df.tcoal.values[i] == minimum_tcoal:
			new_radiusus.append(R)#df.Radius.values[i]*2)
			new_x.append(cdm[0])
			new_y.append(cdm[1])
			new_neighbours.append(d_neigh)
			
			
		else :
			new_radiusus.append(df.Radius.values[i])
			new_x.append(df.x.values[i])
			new_y.append(df.y.values[i])
			new_neighbours.append(df.neighbours.values[i])
		
	
	
	df = df.extract()
	
	
	
	
	
	
	df['Radius'] = np.array(new_radiusus)
	df['x'] = np.array(new_x)
	df['y'] = np.array(new_y)
	df['neighbours'] = np.array(new_neighbours)
	df['xy'] = np.array([[new_x[i],new_y[i]] for i in range(len(new_x))])
	
	
	
	
	
	
	
	 
	#print('Modified DF \n', df)

	# Calculate the new minimum tcoal for the new drop
	# 5 is the position of the neighbour in dataframe
	#all_min_tcoal_with_neighbours(df,n)	
	
	
	df = df.extract()
	df = min_tcoal_singular(df,n)
	'''
	df = df.sort(by=['tcoal'], ascending=[True])
	dr = df[0]
	# Checking the drop is overlapping
	n = n-1
	pos = get_pos_in_df(df,dr[7])
	print('Pos is ',pos)
	if distance_coor(df[pos][4],dr[4]) <= dr[1]:
		coal_event_df(T,df,n)
	print('SIZE OF DF',len(df))
	'''
	

	return df
	
	
def coal_event_df_initial(T,df,n):
	minimum_tcoal = np.array(df.tcoal.values).min()	
	i_list = []
	df_rows = df[df.tcoal.values == minimum_tcoal]
	'''
	for i in range(n):
		if df.tcoal.values[i] == minimum_tcoal:
			i_list.append(i)
	print("ILIST = ",i_list)
	'''
	
	for i in range(n):
		if df.tcoal.values[i] == minimum_tcoal:
			i_list.append(i)
	i_list.append(get_pos_in_df(df,df.Coal_Drop.values[i_list[0]]))
	#i_list.append(get_pos_in_df(df,df.xy.values[i_list[0]]))
	#print("ILIST = ",i_list)
	
	pos1 = i_list[0]
	pos2 = i_list[1]
	R1 = df[pos1][1]
	R2 = df[pos2][1]
	
	
	# Calculate the center of mass coordinates of these two drops
	c1 = df[pos1][4]
	c2 = df[pos2][4]
	cdm = center_of_mass_coor(c1, c2, R1, R2)
	new_point = Point_2(cdm[0],cdm[1])
	
	

	# Calculate the new drops radius
	R =  (R1**3 + R2**3)**(1/3)
	
	
	# Remove the two old drops from the delaunay triangulation and the dataframe
	Del_removal(T,c1[0],c1[1])
	Del_removal(T,c2[0],c2[1])
	# Remove one of the drops from the DF
	
	#df = df[df.x != c1[0]]
	df = df[df.x != c2[0]]
	
	
	#print('ExtractDF',df)
	#print(df.shape)
	
	# Add the new drop to the delaunay triangulation
	Del_add(T,cdm[0],cdm[1])
	
	
	# Calculate the neighbours of the drops
	edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
	d_neigh = data_frame_neighbours(T,new_point,edges)
	
	new_radiusus = []
	new_x = []
	new_y = []
	new_neighbours = []
	new_xy = []
	
	
	
	
	for i in range(len(df)):
		if df.tcoal.values[i] == minimum_tcoal:
			new_radiusus.append(R)#df.Radius.values[i]*2)
			new_x.append(cdm[0])
			new_y.append(cdm[1])
			new_neighbours.append(d_neigh)
			
			
		else :
			new_radiusus.append(df.Radius.values[i])
			new_x.append(df.x.values[i])
			new_y.append(df.y.values[i])
			new_neighbours.append(df.neighbours.values[i])
		
	
	#print('The new radiusus are', new_radiusus)
	df = df.extract()
	
	#print('The new neighbours are', new_neighbours)
	
	
	
	#print('extract DF',df)
	df['Radius'] = np.array(new_radiusus)
	df['x'] = np.array(new_x)
	df['y'] = np.array(new_y)
	df['neighbours'] = np.array(new_neighbours)
	df['xy'] = np.array([[new_x[i],new_y[i]] for i in range(len(new_x))])
	

	# Calculate the new minimum tcoal for the new drop
	# 5 is the position of the neighbour in dataframe
	all_min_tcoal_with_neighbours(df,n)	
	
	
	df = df.extract()
	

	return df
	
	

	
	
	
		

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#	
def data_frame_neighbours(T,point,edges):
	l_neighbours = edge_to_point(get_neighbours(T,point,edges),point) # coordinates of its neighbours
	neighbours =[]
	for l in l_neighbours:
		neighbours.append(l)
	return neighbours
		
	
def drop_radius_from_coord(coord,df): # coord is of type array([x,y])
	new_df = df[df['x'] == coord[0]]
	new_df = df[df['y'] == coord[1]]
	return new_df.Radius.values

		
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def all_min_tcoal_with_neighbours(df,n):
	# 5 is the position of the neighbour in dataframe
	#print('INSERTED DF',df)
	all_min_tcoal = []
	minimum_drop_coords = []
	for i in range(len(df)): # Changeble to just check only the first drop for example
		neigh = df[i][5]
		local_tcoal = []
		#print('NUMBER OF NEIGHBOURS', len(neigh))
		for j in range(len(neigh)):
			# Get the radius of the said neighbouring drop
			coord_neigh_radius = drop_radius_from_coord(neigh[j],df)
			# Get the row of the main drop
			k = df[i]
			neighbour_distance = distance_coor(df[i][4],neigh[j])
			#print('Neighbour_distance',neighbour_distance)
			#print('df[i][1]',df[i][1])
			#print('coord_neigh_radius',coord_neigh_radius)
			tc = (neighbour_distance/(df[i][1]+coord_neigh_radius))**3 * tn[-1]
			#print('TC',tc)
			local_tcoal.append(tc)
			
		# Get the minimum coalescence time for each drop
		local_tcoal = [x for x in local_tcoal if x]
		minimum_tc = min(local_tcoal)
		# Get its index
		minimum_index = local_tcoal.index(minimum_tc)
		minimum_drop_coords.append(neigh[minimum_index])
		all_min_tcoal.append(minimum_tc)
	min_value = min(all_min_tcoal)
	tn.append(min_value)
	df['tcoal'] = np.array(all_min_tcoal)
	df['Coal_Drop'] = np.array(minimum_drop_coords)
	#df['tcoal'] = np.array(listofarray_to_listofscalar(all_min_tcoal))
	#df['tcoal'] = np.array([i[0] for i in all_min_tcoal])
	#df['Coal_Drop'] = np.array(listofarray_to_listofscalar(minimum_drop_coords))
	
	if not isinstance(df.tcoal.values[0], np.floating):
		df['tcoal'] = np.array(listofarray_to_listofscalar(all_min_tcoal))
	
	
	return min_value







def min_tcoal_singular(df,n):
	df = df.sort(by=['tcoal'], ascending=[True])
	dr = df[0]
	neigh = dr[5]
	local_tcoal = []
	minimum_drop_coords = []
	 
	
	for j in range(len(neigh)):
		# Get the radius of the said neighbouring drop
		coord_neigh_radius = drop_radius_from_coord(neigh[j],df)
		# Get the row of the main drop
		neighbour_distance = distance_coor(dr[4],neigh[j])
		#print('Neighbour_distance',neighbour_distance)
		#print('df[i][1]',df[i][1])
		#print('coord_neigh_radius',coord_neigh_radius)
		tc = (neighbour_distance/(dr[1]+coord_neigh_radius))**3 * tn[-1]
		#print('TC',tc)
		local_tcoal.append(tc)
	
	# Get the minimum coalescence time 
	minimum_tc = min(local_tcoal)
	#print('Minimum_tc =', minimum_tc)
	# Get its index
	minimum_index = local_tcoal.index(minimum_tc)
	minimum_drop_coords.append(neigh[minimum_index])
	if minimum_tc == [] :
		minimum_tc = 0
	elif type(minimum_tc) == int : 
		minimum_tc = minimum_tc
	else :
		minimum_tc = minimum_tc[0]	
	
	#print('The minimum drop coords are', minimum_drop_coords)

	dcoal = df.Coal_Drop.values
	dcoal[0] = minimum_drop_coords[0]
	df['Coal_Drop'] = np.array(dcoal)
	
	
	# Checking the drop is overlapping
	if distance_coor(minimum_drop_coords[0],dr[4]) <= dr[1]:
		minimum_tc = 1.7976931348623157e-307
	
	
	
	
	tcoal = df.tcoal.values
	tcoal[0] = minimum_tc
	df['tcoal'] = tcoal
	
	return df
		
		
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#	


     
def mass(r):
    theta= 88 * np.pi / 180
    ftheta = (2 - 3 * np.cos(np.radians(theta)) + (np.cos(np.radians(theta))) ** 3) / (3 * np.sin(np.radians(theta)) ** 3)
    mass = np.pi * ftheta * r ** 3
    return mass

def center_of_mass_coor(c1, c2, r1, r2):
    m1 = mass(r1)
    m2 = mass(r2)
    xc = (m1*c1[0]+m2*c2[0]) / (m1+m2)
    yc = (m1*c1[1]+m2*c2[1]) / (m1+m2)
    return (xc,yc)
	
	

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Function that defines all the drops without coalescence at a time t

def define_all_drops(T,t,r):
	all_drops = []
	index = 0
	for i in T.finite_vertices():
		all_drops.append(drop(index,i.point(),r)) #Start radius at 1
		index +=1
	return all_drops

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



def calculate_surface_area(df):
	area = np.sum(np.pi * (df.Radius.values) ** 2 )
	return area
		


	
	
	

	














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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

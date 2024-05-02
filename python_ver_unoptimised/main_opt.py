import random
import matplotlib.pyplot as plt
import math
from functions_opt import *
from drops_opt import *
from global_variables import *
import time



from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Triangulation_2 import VERTEX
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_circulator
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_handle
from CGAL.CGAL_Kernel import Ref_int
from CGAL.CGAL_Triangulation_2 import Ref_Locate_type_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2

import sys
from functools import lru_cache

sys.setrecursionlimit(15000)
'''
The time for the first DF is 1.8358230590820312
The average time for each coal is 2.9200843631631077
n=700
'''




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
#remove_plot(T,x,y,points)

#Adding the point
add_plot(T,0.5,0.5,points)
'''




num_coal = 15


fig, axs = plt.subplots(2, 2, figsize = (15,15))
t0 = time.time()


df= create_dataframe(T,t0,r)
t_first_df = time.time()
new_df = growth_dataframe_mod(df,n)
first_coal = coal_event_df_initial(T,df,n)
t_first_coal = time.time()


#Creating a automated thing
mean_radius = []
dfs = [df]
dfs2 = [new_df]
dfs3 = [first_coal]
x_axis = []
max_radius = []
min_radius = []
number_drops = [100,99]
#surface_covered = [calculate_surface_area(df),calculate_surface_area(first_coal)]
#500 coalescences
for i in range(num_coal):
	x_axis.append(i)
	n=n-1
	next_growth = growth_dataframe_opt(dfs3[i],n)
	dfs2.append(next_growth)
	
	next_coal = coal_event_df(T,dfs3[i],n)
	dfs3.append(next_coal)
	'''
	Radius3 =  (next_coal.Radius.values) ** 3 
	Radius2 = (next_coal.Radius.values) ** 2
	mean_radius.append((np.sum(Radius3))/(np.sum(Radius2)))
	'''
	mean_radius.append(df.mean(next_coal.Radius))
	max_radius.append(next_coal.max(next_coal.Radius))
	min_radius.append(next_coal.min(next_coal.Radius))
	number_drops.append(len(next_coal))
	#surface_covered.append(calculate_surface_area(next_coal))
	#print(next_coal)
	#bubble_plot5(next_coal)
	
	
	
t2 = time.time()
print('The time for the first DF is', t_first_df - t0)
print('The average time for each coal is', (t2-t_first_df)/(num_coal+1))
print('The time for the first coal is', t_first_coal - t_first_df)
print('THE TOTAL TIME IS',t2-t0)


for i in range(1,3):
	tn[i] = tn[i][0]
t = np.cumsum(tn)


t_func = np.arange(0,0.1,0.001)
t_func2 = np.arange(0,100,0.001)




#Plots
#axs[0,0].plot(t_func,t_func,color = 'red',label = 'y = t')
#axs[0,0].plot(t_func2,t_func2**1/3,color = 'blue',label = 'y = t**1/3')



axs[0,0].scatter(t[:len(mean_radius)],mean_radius,color = 'black',label = 'mean')
axs[0,0].scatter(t[:len(mean_radius)],max_radius,color = 'blue',label ='max')
axs[0,0].scatter(t[:len(mean_radius)],min_radius,color = 'green',label ='min')
axs[0, 0].set_title('Normal Plot')



#axs[0,1].plot(t_func,t_func,color = 'red',label = 'y = t')
#axs[0,1].plot(t_func2,t_func2**1/3,color = 'blue',label = 'y = t**1/3')
#axs[0,1].set_ylim(bottom = 0.001,top = 0.1)
axs[0,1].scatter(t[:len(mean_radius)],mean_radius,color = 'black',label = 'mean')
axs[0,1].scatter(t[:len(mean_radius)],max_radius,color = 'blue',label ='max')
axs[0,1].scatter(t[:len(mean_radius)],min_radius,color = 'green',label ='min')
axs[0,1].set_xscale('log')
axs[0, 1].set_title('X log scale')



#axs[1,0].plot(t_func,t_func,color = 'red',label = 'y = t')
#axs[1,0].plot(t_func2,t_func2**1/3,color = 'blue',label = 'y = t**1/3')
#axs[1,0].set_ylim(bottom = 0.001,top = 0.1)
axs[1,0].scatter(t[:len(mean_radius)],mean_radius,color = 'black',label = 'mean')
axs[1,0].scatter(t[:len(mean_radius)],max_radius,color = 'blue',label ='max')
axs[1,0].scatter(t[:len(mean_radius)],min_radius,color = 'green',label ='min')
axs[1,0].set_yscale('log')
axs[1,0].set_title('Y log scale')



#axs[1,1].plot(t_func,t_func,color = 'red',label = 'y = t')
#axs[1,1].plot(t_func2,t_func2**1/3,color = 'blue',label = 'y = t**1/3')
#axs[1,1].set_ylim(bottom = 0.001,top = 0.1)
axs[1,1].scatter(t[:len(mean_radius)],mean_radius,color = 'black',label = 'mean')
axs[1,1].scatter(t[:len(mean_radius)],max_radius,color = 'blue',label ='max')
axs[1,1].scatter(t[:len(mean_radius)],min_radius,color = 'green',label ='min')
axs[1,1].set_yscale('log')
axs[1,1].set_xscale('log')
axs[1, 1].set_title('log log')



plt.legend()
plt.show()



plt.plot(t[:len(number_drops)],number_drops)
plt.title('n = 100')
plt.ylabel('Number Of Drops')
plt.xlabel('Time')

plt.show()


#plt.plot(t[:len(surface_covered)],surface_covered)
#plt.show()







'''
plt.scatter(x_axis,mean_radius,color = 'black',label = 'mean')
plt.scatter(x_axis,max_radius,color = 'blue',label ='max')
plt.scatter(x_axis,min_radius,color = 'green',label ='min')
plt.legend()
'''



'''
df= create_dataframe(T,t0,r)
mean_radius = []
max_radius = []
min_radius = []



new_df = growth_dataframe_mod(df,n)
print('DATAFRAME AFTER GROWTH')
print(new_df)
print('\n')
#bubble_plot5(new_df)






first_coal = coal_event_df_initial(T,df,n)
print('DATAFRAME AFTER COAL')
print(first_coal)
bubble_plot5(first_coal)


#second_coal_before = growth_dataframe_mod(first_coal,n-1)
#print('SECOND_COAL_BEFORE')
#print(second_coal_before)

#bubble_plot4(second_coal_before)



print('DATAFRAME AFTER SECOND COAL')
second_coal_df = growth_dataframe_opt(first_coal,n-1)
second_coal_df2 = coal_event_df(T,first_coal,n-1)
print(second_coal_df2)
bubble_plot5(second_coal_df2)





print('DATAFRAME AFTER SECONDGROWTH')
third_coal_before = growth_dataframe_opt(second_coal_df2,n-2)
#print(third_coal_before)
#bubble_plot4(third_coal_before)





print('DATAFRAME AFTER THIRDCOAL')
third_coal_df2 = coal_event_df(T,second_coal_df2,n-2)
print(third_coal_df2)
bubble_plot5(third_coal_df2)

print('DATAFRAME AFTER THIRDGROWTH')
fourth_coal_before = growth_dataframe_opt(third_coal_df2,n-3)
#print(fourth_coal_before)
#bubble_plot4(fourth_coal_before)





print('DATAFRAME AFTER FOURTHCOAL')
fourth_coal_df2 = coal_event_df(T,third_coal_df2,n-3)
print(third_coal_df2)
bubble_plot5(fourth_coal_df2)



fifth_coal_before = growth_dataframe_opt(fourth_coal_df2,n-4)
fifth_coal_df2 = coal_event_df(T,fourth_coal_df2,n-4)
bubble_plot5(fifth_coal_df2)

six_coal_before = growth_dataframe_opt(fifth_coal_df2,n-5)
six_coal_df2 = coal_event_df(T,fifth_coal_df2,n-5)
bubble_plot5(six_coal_df2)

seven_coal_before = growth_dataframe_opt(six_coal_df2,n-6)
seven_coal_df2 = coal_event_df(T,six_coal_df2,n-6)
bubble_plot5(seven_coal_df2)

eight_coal_before = growth_dataframe_opt(seven_coal_df2,n-7)
eight_coal_df2 = coal_event_df(T,seven_coal_df2,n-7)
bubble_plot5(eight_coal_df2)


nine_coal_before = growth_dataframe_opt(eight_coal_df2,n-8)
nine_coal_df2 = coal_event_df(T,eight_coal_df2,n-8)





mean_radius.append(df.mean(df.Radius))
mean_radius.append(first_coal.mean(first_coal.Radius))
mean_radius.append(second_coal_df2.mean(second_coal_df2.Radius))
mean_radius.append(third_coal_df2.mean(third_coal_df2.Radius))
mean_radius.append(fourth_coal_df2.mean(fourth_coal_df2.Radius))
mean_radius.append(fifth_coal_df2.mean(fifth_coal_df2.Radius))
mean_radius.append(six_coal_df2.mean(six_coal_df2.Radius))
mean_radius.append(seven_coal_df2.mean(seven_coal_df2.Radius))
mean_radius.append(eight_coal_df2.mean(eight_coal_df2.Radius))
mean_radius.append(nine_coal_df2.mean(nine_coal_df2.Radius))



max_radius.append(df.max(df.Radius))
max_radius.append(first_coal.max(first_coal.Radius))
max_radius.append(second_coal_df2.max(second_coal_df2.Radius))
max_radius.append(third_coal_df2.max(third_coal_df2.Radius))
max_radius.append(fourth_coal_df2.max(fourth_coal_df2.Radius))
max_radius.append(fifth_coal_df2.max(fifth_coal_df2.Radius))
max_radius.append(six_coal_df2.max(six_coal_df2.Radius))
max_radius.append(seven_coal_df2.max(seven_coal_df2.Radius))
max_radius.append(eight_coal_df2.max(eight_coal_df2.Radius))
max_radius.append(nine_coal_df2.max(nine_coal_df2.Radius))



min_radius.append(df.min(df.Radius))
min_radius.append(first_coal.min(first_coal.Radius))
min_radius.append(second_coal_df2.min(second_coal_df2.Radius))
min_radius.append(third_coal_df2.min(third_coal_df2.Radius))
min_radius.append(fourth_coal_df2.min(fourth_coal_df2.Radius))
min_radius.append(fifth_coal_df2.min(fifth_coal_df2.Radius))
min_radius.append(six_coal_df2.min(six_coal_df2.Radius))
min_radius.append(seven_coal_df2.min(seven_coal_df2.Radius))
min_radius.append(eight_coal_df2.min(eight_coal_df2.Radius))
min_radius.append(nine_coal_df2.min(nine_coal_df2.Radius))

x_axis = [i for i in range(len(mean_radius))]
plt.scatter(x_axis,mean_radius,color = 'black')
plt.scatter(x_axis,max_radius,color = 'blue')
plt.scatter(x_axis,min_radius,color = 'green')


'''






























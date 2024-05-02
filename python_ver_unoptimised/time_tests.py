import random
import matplotlib.pyplot as plt
import math
from functions3 import *
from drops import *
import timeit


from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Triangulation_2 import VERTEX
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_circulator
from CGAL.CGAL_Triangulation_2 import Triangulation_2_Vertex_handle
from CGAL.CGAL_Kernel import Ref_int
from CGAL.CGAL_Triangulation_2 import Ref_Locate_type_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2


n=1000

fig, (ax1, ax2) = plt.subplots(1, 2)


# Generate a list of 100 random points in the square (0,0) to (1,1)
points = [(random.random(), random.random()) for i in range(n)]


T = Delaunay_triangulation_2()
points = [Point_2(*p) for p in points]

def insert():
	T.insert(points)

def Del_removal():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T.nearest_vertex(removed_point)
	T.remove(removed_vertex)


numbers = [1,10,100,500,1000,3000,5000,10000]
times = []
times_remove = []
times_remove_max = []
times_reove_min = []

times_max = []
times_min = []
x= 0.5
y= 0.5

'''
for i in range(len(numbers)):
	times.append(timeit.repeat("insert()","from __main__ import insert",repeat = 1,number = numbers[i]))
'''

for i in range(len(numbers)):
	times_remove.append(timeit.repeat("Del_removal()","from __main__ import Del_removal",repeat = 1,number = 1))
	
'''

for i in range(len(times)):
	print(str(numbers[i]))
	print('Standard deviation:', np.std(np.array(times[i])))
	
	


for i in range(len(times)-2):
	ax1.scatter([numbers[i]],np.array(times[i]).mean(),c ='black')
	ax1.scatter([numbers[i]],np.array(times[i]).min(),c ='blue')
	ax1.scatter([numbers[i]],np.array(times[i]).max(),c ='green')
	
'''	

for i in range(len(times)-2):
	ax2.scatter([numbers[i]],np.array(times_remove[i]).mean(),c ='black')
	ax2.scatter([numbers[i]],np.array(times_remove[i]).min(),c ='blue')
	ax2.scatter([numbers[i]],np.array(times_remove[i]).max(),c ='green')

'''
	
ax1.scatter([numbers[len(times)-1]],np.array(times[len(times)-1]).mean(),c ='black',label = 'Mean times')
ax1.scatter([numbers[len(times)-1]],np.array(times[len(times)-1]).min(),c ='blue',label= 'Min times')
ax1.scatter([numbers[len(times)-1]],np.array(times[len(times)-1]).max(),c ='green',label='Max times')	
'''
ax2.scatter([numbers[len(times_remove)-1]],np.array(times_remove[len(times)-1]).mean(),c ='black',label = 'Mean times')
ax2.scatter([numbers[len(times_remove)-1]],np.array(times_remove[len(times)-1]).min(),c ='blue',label= 'Min times')
ax2.scatter([numbers[len(times_remove)-1]],np.array(times_remove[len(times)-1]).max(),c ='green',label='Max times')	


'''
ax1.title('Delaunay Triangulation Insert Times')
ax1.legend()
ax1.xlabel('Iterations')
ax1.ylabel('Time(s)')
'''
ax2.title('Delaunay Triangulation Remove Times')
ax2.legend()
ax2.xlabel('Iterations')
ax2.ylabel('Time(s)')


plt.show()

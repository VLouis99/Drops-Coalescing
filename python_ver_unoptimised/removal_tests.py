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


n1=10000
n2 = 50000
n3 =100000
n4 = 1000000
n5=500000
n6 = 1250000
n7 = 1750000



# Generate a list of 100 random points in the square (0,0) to (1,1)
points = [(random.random(), random.random()) for i in range(n1)]
points2 = [(random.random(), random.random()) for i in range(n2)]
points3 = [(random.random(), random.random()) for i in range(n3)]
points4 = [(random.random(), random.random()) for i in range(n4)]
points5 = [(random.random(), random.random()) for i in range(n5)]
points6 = [(random.random(), random.random()) for i in range(n6)]
points7 = [(random.random(), random.random()) for i in range(n7)]

T1 = Delaunay_triangulation_2()
T2 = Delaunay_triangulation_2()
T3 = Delaunay_triangulation_2()
T4 = Delaunay_triangulation_2()
T5 = Delaunay_triangulation_2()
T6 = Delaunay_triangulation_2()
T7 = Delaunay_triangulation_2()

points = [Point_2(*p) for p in points]
points2 = [Point_2(*p) for p in points2]
points3 = [Point_2(*p) for p in points3]
points4 = [Point_2(*p) for p in points4]
points5 = [Point_2(*p) for p in points5]
points6 = [Point_2(*p) for p in points6]
points7 = [Point_2(*p) for p in points7]

T1.insert(points)
T2.insert(points2)
T3.insert(points3)
T4.insert(points4)
T5.insert(points5)
T6.insert(points6)
T7.insert(points7)


def Del_removal():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T1.nearest_vertex(removed_point)
	T1.remove(removed_vertex)



def Del_removal1():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T2.nearest_vertex(removed_point)
	T2.remove(removed_vertex)
	

def Del_removal2():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T3.nearest_vertex(removed_point)
	T3.remove(removed_vertex)
	

def Del_removal3():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T4.nearest_vertex(removed_point)
	T4.remove(removed_vertex)
	
def Del_removal5():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T5.nearest_vertex(removed_point)
	T5.remove(removed_vertex)
	
def Del_removal6():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T6.nearest_vertex(removed_point)
	T6.remove(removed_vertex)
	
def Del_removal7():
	x = points[0].x()
	y = points[0].y()
	removed_point = Point_2(x,y)
	removed_vertex= T7.nearest_vertex(removed_point)
	T7.remove(removed_vertex)
	
	

times = []
times_max = []
times_min = []

times1 = []
times_max1 = []
times_min1 = []

times2 = []
times_max2 = []
times_min2 = []

times3 = []
times_max3 = []
times_min3 = []

times5 = []
times_max5 = []
times_min5 = []

times6 = []
times_max6 = []
times_min6 = []

times7 = []
times_max7 = []
times_min7 = []

x= 0.5
y= 0.5


times.append(timeit.repeat("Del_removal()","from __main__ import Del_removal",repeat = 5,number = 1))


times1.append(timeit.repeat("Del_removal1()","from __main__ import Del_removal1",repeat = 5,number = 1))
times2.append(timeit.repeat("Del_removal2()","from __main__ import Del_removal2",repeat = 5,number = 1))

times3.append(timeit.repeat("Del_removal3()","from __main__ import Del_removal3",repeat = 5,number = 1))

times5.append(timeit.repeat("Del_removal5()","from __main__ import Del_removal5",repeat = 5,number = 1))

times6.append(timeit.repeat("Del_removal6()","from __main__ import Del_removal6",repeat = 5,number = 1))

times5.append(timeit.repeat("Del_removal7()","from __main__ import Del_removal7",repeat = 5,number = 1))




plt.scatter(n1,np.array(times).mean(),c ='black',label = 'Mean times')
plt.scatter(n1,np.array(times).min(),c ='blue',label= 'Min times')
plt.scatter(n1,np.array(times).max(),c ='green',label='Max times')	

plt.scatter(n2,np.array(times1).mean(),c ='black')
plt.scatter(n2,np.array(times1).min(),c ='blue')
plt.scatter(n2,np.array(times1).max(),c ='green')	


plt.scatter(n3,np.array(times2).mean(),c ='black')
plt.scatter(n3,np.array(times2).min(),c ='blue')
plt.scatter(n3,np.array(times2).max(),c ='green')	


plt.scatter(n4,np.array(times3).mean(),c ='black')
plt.scatter(n4,np.array(times3).min(),c ='blue')
plt.scatter(n4,np.array(times3).max(),c ='green')	

plt.scatter(n5,np.array(times5).mean(),c ='black')
plt.scatter(n5,np.array(times5).min(),c ='blue')
plt.scatter(n5,np.array(times5).max(),c ='green')	

plt.scatter(n6,np.array(times5).mean(),c ='black')
plt.scatter(n6,np.array(times5).min(),c ='blue')
plt.scatter(n6,np.array(times5).max(),c ='green')	

plt.scatter(n7,np.array(times5).mean(),c ='black')
plt.scatter(n7,np.array(times5).min(),c ='blue')
plt.scatter(n7,np.array(times5).max(),c ='green')	




plt.title('Removal Times')
plt.legend()
plt.xlabel('n')
plt.ylabel('Time(s)')


plt.show()

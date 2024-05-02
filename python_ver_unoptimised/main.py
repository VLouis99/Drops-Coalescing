
import sys

sys.path.append('scr')
from first_test import *


'''
Initialisations
'''
n=20
L = total_drops(n) # generating n total drops 




'''
Calculating Delaunay Triangulation
'''
T = Delaunay_triangulation_2()
vp1 = Point_2(rd.uniform(-0.5, 0.5),rd.uniform(-0.5, 0.5)) # adding one extra point
#Inserting each point one by one
v1 = T.insert(vp1)
'''
lt=Ref_Locate_type_2()
print(lt)	
c = T.locate(vp1, lt)
'''

# Extract the edges of the triangulation
edges = [ (T.segment(e).source(), T.segment(e).target()) for e in T.finite_edges()]
print('edges ---------- ' , edges)


#------------------
#removing and adding testing
#------------------



v_handles = calc_triang_draw2(T,L,(0,0)) # First plot without removal 
''' 
optimisation
remove only the concerned edges instead of redrawing all the other edges 
but this would have to use a where ----> n complexity in all cases???
'''
removed_handle = remove(T,L,v_handles,4,(0,1)) # removed handle is the handle that has been removed, used to replot in red the removed point

v_handles2 = calc_triang_draw2(T,L,(0,1)) # second plot after removal of handle

added_vertex = add(T,L,(1,0))
#draw_singular_connected_edges(T,L,added_vertex,(1,0))

v_handles3 = calc_triang_draw2(T,L,(1,0))

#draw_droplets(L) # Drawing the droplets




#draw_singular_connected_edges(T,v1)
#plt.scatter(vp1.x(),vp1.y(),s=30,color = 'red')




'''
list_handles = []
for i in L:
	w = T.insert(i)
	list_handles.append(w)
	
	list_del_vertex = connected_vertexs(T,w) 
	for j in list_del_vertex : 
		draw_edge(i,j) # optimisable
'''	




assert T.is_valid()




print("Nb vertices ", T.number_of_vertices())






'''
Find all the points to which a selected vertex is connected to through delaunay triangularisation
here for v1
'''
'''
circulator = T.incident_vertices(v1) # starting at the point v1
done = circulator.next() # going to the next point
list_del_points = [] # list of linked points to the point considered

while(1):
  v = circulator.next()
  print('V type is ', type(v) )
  vp = v.point() # vp is the point associated to the vertex v on the circulator
  list_del_points.append((vp.x(),vp.y()))
  if v == done:
    break
'''



#list_del_points = connected_points(T,v1) # list of delaunays connected points of one point v1
#code to plot	
#pts = extract(L)

#connected_points = scatter_convert(list_del_points)
#print('list_del_points= --------------', list_del_points)
#plt.scatter(pts[0],pts[1],s=10)
#plt.scatter(vp1.x(),vp1.y(),s=15)
#plt.scatter(connected_points[0],connected_points[1],s=10)

'''
list_del_vertex = connected_vertexs(T,v1)

for i in list_del_vertex : 
	draw_edge(vp1,i)
'''

#print(scatter_convert(list_del_points))

#draw_all_edges(T,L)

#plt.xlim(vp1.x()-0.1,vp1.x()+0.1)
#plt.ylim(vp1.y()-0.1,vp1.y()+0.1)


'''
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(pts[0],pts[1],s=10)
ax2.scatter(pts[0],pts[1],s=10)
'''











plt.show()
		

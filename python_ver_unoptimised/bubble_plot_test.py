
import numpy as np
import vaex
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
x = np.array([0])
y=np.array([0])
radius = np.array([10])
indices = np.array([0])
df_test = vaex.from_arrays(Indices = indices, Radius = radius , x = x ,y = y ) 
color = np.random.rand(1)
cmap = plt.cm.hsv


# df is a tuple
def bubble_plot5(df):
	size = radius
	f = plt.figure(figsize=(7, 7), dpi=100)
	ax = f.add_subplot(1,1,1)
	

	offsets = list(zip(x, y))
	ax.add_collection(EllipseCollection(widths=size*2, heights=size*2, angles=0, units='xy',
		                                   facecolors=plt.cm.hsv(color),
		                                   offsets=offsets, transOffset=ax.transData))
	plt.xlim(-10,10)
	plt.ylim(-10,10)
	



bubble_plot5(df_test)
plt.show()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

f = plt.figure(figsize=(7, 7), dpi=100)
ax = f.add_subplot(1,1,1)
ax.set_ylim(-105,105)
ax.set_xlim(-105,105)

x =[0]
y=[0]
L = 14.0 # Diameter of a circle in data units 
# marker_radius is L/2 in points.
marker_radius = (ax.transData.transform((0,0))
                 -ax.transData.transform((L/2,0)))[0]
marker_area = 2.3*marker_radius**2
ax.scatter(x, y, color='#7fc97f', edgecolors='None', s=marker_area)

plt.show()

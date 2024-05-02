import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



x = np.linspace(0,10000)

y = x
y3 = x**(1/3)


df_time = pd.read_csv("time.csv")
df_radius = pd.read_csv("radius.csv")


plt.plot(df_time.values[:len(df_radius.values)],df_radius.values)


#plt.plot(x,y*min(df_radius.values))
plt.plot(x,y3*min(df_radius.values))

#plt.yscale('log')
plt.xscale('log')

plt.show()

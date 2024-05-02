import vaex 
import random as random
import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt

numbers = [100000000]
n = 500000

points = []
dfs = []
xl = []
yl = []
for i in range(len(numbers)):
	pointsx = [random.random() for j in range(numbers[i])]
	pointsy = [random.random() for j in range(numbers[i])]
	x = np.array(pointsx)
	y = np.array(pointsy)
	array =[x,y]
	xl.append(x)
	yl.append(y)
	dfs.append(vaex.from_arrays(x = x, y = y))

	
	





#array = [x,y]

#df = vaex.from_arrays(x = x, y = y)
times = []

def create_data_frame(x,y):
	df = vaex.from_arrays(x = x, y = y)
	return df




def create_pandas_frame():
	df_pandas = pd.DataFrame({'x': xl[-1], 'y': y[-1]})
	return df_pandas

df_pandas = create_pandas_frame()


def change_entire_column(df):
	df['x'] = df['x'] + 10
	
def change_entire_column_pandas():
	df_pandas['x'] = df_pandas['x']+ 10 
	
def df_sort():
	df2 = df.sort(by=['x'], ascending=[True])


def apply_func_for():
	for i in range(len(df)):
		df.x.values[i] = 2*df.x.values[i]
	return df


def df_pandas_sort():
	return df_pandas.sort_values('x')
	
def vaex_mean():
	return df.mean(df.x)
	
def pandas_mean():
	df_pandas['x'].mean()
	

t_create_vaexs = []
t_modify_vaexs = []
for i in range(len(numbers)):
	t_create_vaex = timeit.Timer(create_data_frame(xl[i],yl[i]),"from __main__ import create_data_frame" )
	t_modify_vaex  = timeit.Timer(create_data_frame(dfs[i],dfs[i]),"from __main__ import change_entire_column" )
	
	t_create_vaexs.append(np.mean(t_create_vaex.repeat(1)))
	t_modify_vaexs.append(np.mean(t_modify_vaex.repeat(1)))
	


plt.scatter(numbers[-1],timeit.Timer("create_pandas_frame()","from __main__ import create_pandas_frame").timeit(1),label='Pandas_create')
plt.scatter(numbers[-1],timeit.Timer("change_entire_column_pandas()","from __main__ import change_entire_column_pandas").timeit(1),label='Pandas_modify')
plt.scatter(numbers,t_create_vaexs,label='Create')
plt.scatter(numbers,t_modify_vaexs,label='Modify')

plt.legend()
plt.show()
	
'''
df_pandas = create_pandas_frame()
apply_func_for()

t_create_vaex = timeit.Timer("create_data_frame()", "from __main__ import create_data_frame")
t_create_pandas = timeit.Timer("create_pandas_frame()", "from __main__ import create_pandas_frame")

t = timeit.Timer("change_entire_column()", "from __main__ import change_entire_column")
#tapply = timeit.Timer("apply_func_for()", "from __main__ import apply_func_for")
tsort = timeit.Timer("df_sort()", "from __main__ import df_sort")
tpandas = timeit.Timer("df_pandas_sort()", "from __main__ import df_pandas_sort")
tchangepandas = timeit.Timer("change_entire_column_pandas()", "from __main__ import change_entire_column_pandas")



vaex_meant = timeit.Timer("vaex_mean()", "from __main__ import vaex_mean")
pandas_meant = timeit.Timer("pandas_mean()", "from __main__ import pandas_mean")


timecreate = t_create_vaex.timeit(1)
time_pandas_create = t_create_pandas.timeit(1)

time = t.timeit(1)
tchangepandas = tchangepandas.timeit(1)

#timetapply = tapply.timeit(1)

timesort = tsort.timeit(1)
timesortpandas = tpandas.timeit(1)

time_vaexmean = vaex_meant.timeit(1)
time_pandasmean = pandas_meant.timeit(1)


print('Time to change entire column in Vaex,Pandas',time,tchangepandas)
print('Time for creation',timecreate,time_pandas_create)
#print('Time for the apply_func_for',timetapply)
print('Time for the sorts, vaex and pandas ',timesort,timesortpandas)


print('Time mean, vaex and pandas', time_vaexmean,time_pandasmean)
'''



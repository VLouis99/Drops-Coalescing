import vaex 
import random as random
import numpy as np
import timeit
import pandas as pd
from memory_profiler import memory_usage

n = 500000

pointsx = [random.random() for i in range(n)]
pointsy = [random.random() for i in range(n)]


x = np.array(pointsx)
y = np.array(pointsy)


array = [x,y]

df = vaex.from_arrays(x = x, y = y)
times = []





def create_data_frame():
	df = vaex.from_arrays(x = x, y = y)
	return df


def create_pandas_frame():
	df_pandas = pd.DataFrame({'x': x, 'y': y})
	return df_pandas


def change_entire_column():
	df['x'] = df['x'] + 10
	
def change_entire_column_pandas():
	df_pandas['x'] = df_pandas['x']+ 10 
	
def df_sort():
	df2 = df.sort(by=['x'], ascending=[True])
	return df_sort()


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
	

df_pandas = create_pandas_frame()


mem_usage = memory_usage((create_data_frame,))
print(f'Memory used by function: {mem_usage[0]} MiB')
mem_usage_pandas = memory_usage((create_pandas_frame,))
print(f'Memory used by function: {mem_usage_pandas[0]} MiB')


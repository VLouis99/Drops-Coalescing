import matplotlib.pyplot as plt
import csv
import numpy as np
import os


times = []
means =[]
mins = []
maxs = []
surfacecovered = []
vertices = []

import csv
with open('timee.txt', 'r') as fd:
    reader = csv.reader(fd)
    for row in reader:
        times.append(float(row[0]))



with open('meane.txt', 'r') as fd2:
    reader = csv.reader(fd2)
    for row in reader:
    	means.append(float(row[0]))
    
        
with open('mine.txt', 'r') as fd2:
    reader = csv.reader(fd2)
    for row in reader:
    	mins.append(float(row[0]))
    
        
with open('maxe.txt', 'r') as fd2:
    reader = csv.reader(fd2)
    for row in reader:
    	maxs.append(float(row[0]))
    
    
with open('surfacecovered.txt', 'r') as fd2:
    reader = csv.reader(fd2)
    for row in reader:
    	surfacecovered.append(float(row[0]))
    	
with open('n_evo.txt', 'r') as fd2:
    reader = csv.reader(fd2)
    for row in reader:	
    	vertices.append(float(row[0]))
        






t = np.linspace(1,500)
tt = np.linspace(730,10000)




'''
# Plot of Number of Drops vs Time
plt.title('Number of Drops vs Time')
plt.scatter(times[:len(vertices)],np.array(vertices),c = 'Blue')
plt.xlabel('Time')
plt.ylabel('Number of Drops')
plt.yscale('log')
plt.xscale('log')
plt.show()
'''


# Plot of Radius vs Time
plt.plot(tt,tt*means[0]*0.01,c='hotpink',lw= 2 , label = 't')
plt.plot(t,t**(1.0/3.0)*means[0],c ='red', lw = 2 ,label = 't^1/3')

plt.scatter(times[:len(means)],means[:],marker = 'x',label = 'means')
plt.scatter(times[:len(means)],maxs[:],marker ='o',label = 'maxs')
plt.scatter(times[:len(means)],mins[:],label = 'mins')



plt.xlabel('Time')
plt.ylabel('Mean Radius')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.title('Evolution of Radius')
plt.show()
'''
# Plot of number of drops vs time
plt.scatter(times[:len(vertices)],vertices)
plt.title('Number of drops vs Time')
plt.xlabel('Time')
plt.ylabel('n')
plt.xscale('log')
plt.yscale('log')

plt.show()


# Plot of Drops vs Radius
plt.title('Number of drops vs Mean Radius')
plt.xlabel('<R>')
plt.ylabel('n')
plt.scatter(means,vertices)
plt.xscale('log')
plt.yscale('log')
plt.show()

# Plot of Surface vs Time
plt.title('Surface vs Time')
plt.scatter(times[:len(surfacecovered)],np.array(surfacecovered)/4,c = 'Blue')
plt.xlabel('Time')
plt.ylabel('Surface covered')
plt.yscale('log')
plt.xscale('log')
plt.show()


# Load and plot histograms
hist_dir = '/home/anamaria/Documents/'  # Directory containing histogram text files

for time in times:
    hist_file = os.path.join(hist_dir, f'histogram_{int(time)}.txt')  # Adjust the filename format

    # Check if the histogram file exists before attempting to open it
    if os.path.isfile(hist_file):
        data = []
        with open(hist_file, 'r') as hist_fd:
            reader = csv.reader(hist_fd, delimiter=' ')
            for row in reader:
                data.append([float(row[0]), int(row[1])])

        data = sorted(data, key=lambda x: x[0])
        x_values = [item[0] for item in data]
        y_values = [item[1] for item in data]

        # Plot the histogram
        plt.figure()
        plt.bar(x_values, y_values, width=0.8 * (x_values[1] - x_values[0]), align='center', alpha=0.7)
        plt.title(f'Histogram at t = {time}')
        plt.xlabel('Radius (r)')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()
'''












# Project Title: Drop Coalescence Analysis

## Overview
This project analyzes the coalescence of water droplets on a slanted surface. The study focuses on the evolution of drop radii due to condensation, examining how drop size changes over time, closely following theoretical predictions (\(t^{1/3}\) and \(t\) laws).

## Results
Our findings, visualized in the included plots, show the predicted versus actual growth rates of water droplets under controlled conditions. Key observations include variations in droplet size due to differing environmental factors and droplet coalescence rates.

![Evolution of Radius](/Git_Projects/Drops_Coal/Drops-Coalescing/evolution_of_radius.png)

## Methodology
- **Data Collection**: Measurements were taken of droplet sizes over time using high-resolution imaging techniques.
- **Data Analysis**:
  - **Triangulation**: Delaunay triangulation was employed to optimize the spatial analysis of droplets.
  - **Computational Analysis**: Initial computations were performed using Python libraries Vaex and Pandas for data handling and analysis. Due to performance limitations, critical components of the data processing were transitioned to C++ to enhance computational efficiency.

## Performance Comparison

To optimize our data processing, we evaluated the performance of two popular Python libraries, Vaex and Pandas. The comparison focused on both execution time and memory usage for several common operations essential to our analysis:

### Execution Time

| Function    | Vaex     | Pandas  |
|-------------|----------|---------|
| Creation    | 5.04 ms  | 3.00 ms |
| Modifying   | 0.3 ms   | 5.8 ms  |
| Sort        | 8.1 ms   | 6 ms    |
| Mean        | 9 ms     | 1 ms    |

### Memory Usage

| Function    | Vaex          | Pandas         |
|-------------|---------------|----------------|
| Creation    | 215.36 MiB    | 223.02 MiB     |
| Modifying   | 216.78 MiB    | 219.87 MiB     |
| Sort        | 221.21 MiB    | 237.31 MiB     |
| Mean        | 215.77 MiB    | 216.45 MiB     |

## Graphical Results

This section presents key graphical outputs from our analysis, illustrating the dynamics of droplet coalescence and surface coverage over time.

### Evolution of Radius
The first graph depicts the mean, maximum, and minimum radii of droplets over time. The data follows a predictable growth pattern, aligning with the theoretical models (\(t^{1/3}\) and \(t\)) of droplet growth.

### Surface Coverage Over Time
The second graph shows the progression of the surface area covered by droplets as a function of time, demonstrating a consistent increase in coverage before plateauing, indicating a saturation point.

### Number of Drops vs. Time and Mean Radius
The third graph highlights the relationship between the number of drops and time, and the mean radius of drops over time. It illustrates the decrease in droplet count due to coalescence, alongside the increase in mean radius, a critical aspect of our study for understanding droplet dynamics on slanted surfaces.

![Evolution of Radius](/Git_Projects/Drops_Coal/Drops-Coalescing/evolution_of_radius.png)
![Surface vs Time](/Git_Projects/Drops_Coal/Drops-Coalescing/surface_vs_time.png)
![Number of Drops vs Time](/Git_Projects/Drops_Coal/Drops-Coalescing/number_of_drops_vs_time.png)

## Technologies Used
- Python: Pandas, Vaex
- C++ for performance-critical data processing

## How to Run the Code
Include instructions on how to set up and run your project. This may involve:
```bash
git clone https://your-repository-link
cd your-project-folder
./compile_and_run_instructions


import matplotlib.pyplot as plt
import numpy as np

# List of dictionaries as provided
data = [
    {1: 2.3459, 2: 1.1561, 3: 2.3944},
    {1: 2.7572, 2: 3.9885, 3: 1.9448},
    {1: 7.7927, 2: 5.063, 3: 5.2545},
    {1: 5.9533, 2: 5.6108, 3: 4.1819},
    {1: 5.2079, 2: 5.3276, 3: 4.2879},
    {1: 4.1365, 2: 4.1321, 3: 3.8574},
    {1: 3.5832, 2: 4.9827, 3: 3.9041}
]

# Convert the list of dictionaries into a list of lists, ensuring order by keys
data_list = [[d[k] for k in sorted(d)] for d in data]

# Create a NumPy array from the list of lists
flow = np.array(data_list).T
density = np.array([0.5, 1, 2, 3, 4, 5, 6])
# print(flow[0].shape, density.shape)

# Create a scatter plot
plt.scatter(density, flow[0], color='b', marker='o', label='Flow at measuring point 1')
plt.scatter(density, flow[1], color='r', marker='+', label='Flow at measuring point 2')
plt.scatter(density, flow[2], color='g', marker='*', label='Flow at measuring point 3')
# Customize the plot (add labels and title)
plt.xlabel('Density (P/m^2)')
plt.ylabel('Flow (P/m/s)')
plt.title('Measured flow w.r.t density at 3 measuring points')
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig('../outputs/measured_flow_density.png', dpi=300)
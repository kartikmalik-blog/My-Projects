import numpy as np

# 1. Create a 3x4 matrix (3 rows, 4 columns) of random data (0 to 99)
data_matrix = np.random.randint(0, 100, size=(3, 4))
print("Original Matrix (3x4):\n", data_matrix)

#2. Find the maximum value in each column
max_per_column = np.max(data_matrix, axis=0)
print("\nMax Value per Column (axis=0):\n", max_per_column)

#3. Calculate the mean of the entire matrix
overall_mean = np.mean(data_matrix)
print("\nOverall Mean:\n", overall_mean)
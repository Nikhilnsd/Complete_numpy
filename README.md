# 📚 Complete Numpy Tutorial

Welcome to the **Numpy Tutorial**! This project showcases fundamental and advanced operations using Numpy, a powerful numerical computing library in Python. The tutorial covers array creation, manipulation, mathematical operations, indexing, slicing, and performance benchmarks.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.x
- Install Numpy: `pip install numpy`

### Running the Code
Run the code examples in:
- A Python interpreter
- A Jupyter Notebook for interactive exploration

---

## 📑 Table of Contents

1. [Array Creation](#array-creation)
2. [Basic Operations](#basic-operations)
3. [Indexing and Slicing](#indexing-and-slicing)
4. [Array Manipulation](#array-manipulation)
5. [Mathematical Operations](#mathematical-operations)
6. [Iterating Through Arrays](#iterating-through-arrays)
7. [Performance Comparison](#performance-comparison)
8. [Advanced Numpy](#advanced-numpy)

---

## 🌀 Array Creation

Create arrays of various dimensions and special types.

### import numpy as np

# 1D Array
a = np.array([71, 89, 45])
print("1D Array:\n", a)

# 2D Array
b = np.array([[71, 89, 45], [71, 85, 56]])
print("2D Array:\n", b)

# 3D Array
c = np.array([[[7, 8], [3, 2]], [[4, 5], [7, 6]]])
print("3D Array:\n", c)

# Special Arrays
zeros_array = np.zeros((2, 3))
ones_array = np.ones((3, 3))
identity_matrix = np.eye(3)
random_array = np.random.random((3, 3))
print("Zeros Array:\n", zeros_array)
print("Ones Array:\n", ones_array)
print("Identity Matrix:\n", identity_matrix)
print("Random Array:\n", random_array)

# Reshaping and Range
reshaped = np.arange(12).reshape(3, 4)
print("Reshaped Array:\n", reshaped)

# Output Example:

1D Array:
 [71 89 45]
2D Array:
 [[71 89 45]
 [71 85 56]]
3D Array:
 [[[7 8]
  [3 2]]
 [[4 5]
  [7 6]]]
Zeros Array:
 [[0. 0. 0.]
 [0. 0. 0.]]
Ones Array:
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
Identity Matrix:
 [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Random Array:
 [[0.43316523 0.0674609  0.84757491]
 [0.0891274  0.70556089 0.50350373]
 [0.57478671 0.90763548 0.61141924]]
Reshaped Array:
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]


### Prerequisites
- Python 3.x
- Install Numpy: `pip install numpy`

### Running the Code
Run the code examples in:
- A Python interpreter
- A Jupyter Notebook for interactive exploration


## 🌀 Array Creation

Create arrays of various dimensions and special types.

import numpy as np

# 1D Array
a = np.array([71, 89, 45])
print("1D Array:\n", a)

# 2D Array
b = np.array([[71, 89, 45], [71, 85, 56]])
print("2D Array:\n", b)

# 3D Array
c = np.array([[[7, 8], [3, 2]], [[4, 5], [7, 6]]])
print("3D Array:\n", c)

# Special Arrays
zeros_array = np.zeros((2, 3))
ones_array = np.ones((3, 3))
identity_matrix = np.eye(3)
random_array = np.random.random((3, 3))
print("Zeros Array:\n", zeros_array)
print("Ones Array:\n", ones_array)
print("Identity Matrix:\n", identity_matrix)
print("Random Array:\n", random_array)

# Reshaping and Range
reshaped = np.arange(12).reshape(3, 4)
print("Reshaped Array:\n", reshaped)
Output Example:

lua
Copy code
1D Array:
 [71 89 45]
2D Array:
 [[71 89 45]
 [71 85 56]]
3D Array:
 [[[7 8]
  [3 2]]
 [[4 5]
  [7 6]]]
Zeros Array:
 [[0. 0. 0.]
 [0. 0. 0.]]
Ones Array:
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
Identity Matrix:
 [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Random Array:
 [[0.43316523 0.0674609  0.84757491]
 [0.0891274  0.70556089 0.50350373]
 [0.57478671 0.90763548 0.61141924]]
Reshaped Array:
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
## ➕ Basic Operations
### Perform basic mathematical operations on arrays.

### Mathematical Functions
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
print("Addition:\n", a + b)
print("Multiplication:\n", a * b)

### Broadcasting
c = np.array([[1], [2], [3]])
broadcast_result = c + np.array([10, 20, 30])
print("Broadcasting Result:\n", broadcast_result)

### Output Example:

Addition:
 [11 22 33]
Multiplication:
 [10 40 90]
Broadcasting Result:
 [[11 21 31]
 [12 22 32]
 [13 23 33]]

🔍 Indexing and Slicing
### Extract specific elements or subarrays from an array.

### Indexing
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("Element at [1, 2]:", arr[1, 2])

### Slicing
print("First two rows:\n", arr[:2, :])
print("Last column:\n", arr[:, -1])

### Advanced Indexing
indices = [0, 2]
print("Selected rows:\n", arr[indices])


### Output Example:

Element at [1, 2]: 60
First two rows:
 [[10 20 30]
 [40 50 60]]
Last column:
 [30 60 90]
Selected rows:
 [[10 20 30]
 [70 80 90]]

## 🔄 Array Manipulation
### Manipulate arrays through stacking, splitting, and transposing.

### Stacking
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
h_stack = np.hstack((arr1, arr2))
v_stack = np.vstack((arr1, arr2))
print("Horizontal Stack:\n", h_stack)
print("Vertical Stack:\n", v_stack)

### Splitting
split_result = np.split(arr1, 2, axis=0)
print("Split Result:\n", split_result)

### Transpose
print("Transpose of arr1:\n", arr1.T)


### Output Example:

Horizontal Stack:
 [[1 2 5 6]
 [3 4 7 8]]
Vertical Stack:
 [[1 2]
 [3 4]
 [5 6]
 [7 8]]
Split Result:
 [array([[1, 2]]), array([[3, 4]])]
Transpose of arr1:
 [[1 3]
 [2 4]]


## 🧮 Mathematical Operations
### Perform dot products, trigonometric, exponential, and logarithmic operations.

### Dot Product
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
dot_product = np.dot(a, b)
print("Dot Product:\n", dot_product)

### Trigonometric Functions
angles = np.array([0, np.pi / 2, np.pi])
sin_values = np.sin(angles)
print("Sine Values:\n", sin_values)

### Exponentials and Logarithms
exponential = np.exp(a)
logarithm = np.log(np.array([1, 2, 3]))
print("Exponential:\n", exponential)
print("Logarithm:\n", logarithm)

### Output Example:

Dot Product:
 [[19 22]
 [43 50]]
Sine Values:
 [0. 1. 0.]
Exponential:
 [[ 2.71828183  7.3890561 ]
 [20.08553692 54.59815003]]
Logarithm:
 [0.         0.69314718 1.09861229]


## 🔁 Iterating Through Arrays
### Iterate over arrays using loops and np.nditer.

arr = np.array([[1, 2], [3, 4]])
print("Iterating through array:")
for row in arr:
    print(row)

### Using nditer
print("Iterating with np.nditer:")
for x in np.nditer(arr):
    print(x)

### Output Example:

Iterating through array:
[1 2]
[3 4]
Iterating with np.nditer:
1
2
3
4


## ⚡ Performance Comparison
### Compare performance between Python lists and Numpy arrays.

import time

### Python Lists
a = [i for i in range(10000000)]
b = [i for i in range(10000000, 20000000)]
start = time.time()
c = [a[i] + b[i] for i in range(len(a))]
print("Python List Time:", time.time() - start)

### Numpy Arrays
a_np = np.arange(10000000)
b_np = np.arange(10000000, 20000000)
start = time.time()
c_np = a_np + b_np
print("Numpy Array Time:", time.time() - start)

### Output Example:

Python List Time: 1.23456 seconds
Numpy Array Time: 0.12345 seconds


## 🚀 Advanced Numpy
## Speed Comparisons:
### Numpy is faster due to optimized C-extensions.

## Efficient Computations:
### Use broadcasting and vectorization for better performance.

#### Broadcasting Example
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
result = a + b
print("Broadcasted Addition:\n", result)

### Efficient Computation
large_array = np.random.random((1000, 1000))
sum_result = np.sum(large_array, axis=0)
print("Sum along axis=0:\n", sum_result)
``


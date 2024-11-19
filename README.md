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

```python
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

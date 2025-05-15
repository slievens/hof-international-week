---
title: Einops and einsum
subtitle: International Week @ Hof
author: Stijn Lievens
date: May 2025
---

# Einops and einsum

## What is einsum?

- **Einsum** is a function in Numpy, Keras and Pytorch that allows you to perform tensor operations using **Einstein summation convention**.
- We will use the implementation in the library `einops` because it is more readable.
  - `einops` also provides additional functions to manipulate tensors.

## Einstein summation convention

The formula for matrix multiplication is:
$$
\mathbf{C}_{ij} = \sum_{k} \mathbf{A}_{ik} \mathbf{B}_{kj}
$$
With the Einstein summation convention:
$$
\mathbf{C}_{ij} = \mathbf{A}_{i\textcolor{red}{k}} \mathbf{B}_{\textcolor{red}{k}j}
$$

> Since the index $k$ is repeated in the input, it is summed over. This is the **Einstein summation convention**.

## Example in Python

Remember: $\mathbf{C}_{ij} = \mathbf{A}_{i\textcolor{red}{k}} \mathbf{B}_{\textcolor{red}{k}j}$

```python
A = np.arange(12).reshape(3,4)          # 3 by 4 matrix
B = np.arange(-4,4).reshape(4,2)        # 4 by 2 matrix
C = einops.einsum(A, B, "i k,k j->i j") # Spaces needed with einops
```
gives the same result as:
```python
C = np.empty(shape=(3,2))
for i in range(3):      # First outer loop
    for j in range(2):  # Second outer loop
        total = 0       # Start of inner loop
        for k in range(4):  
            total += A[i, k]*B[k, j]
        C[i,j] = total  # End of inner loop. Assign total to C[i,j]
```        

## General Rules

- **Free indices**: indices that appear in the output.
  - $i$ and $j$ are free indices in the example above.
- **Summation indices**: indices that appear in the input but not in the output.
  - $k$ is a summation index in the example above.

- The free indices are not summed over. 
  - These are in the "outer" loops.
- The summation indices are summed over. 
  - These are in the "inner" loops.

## Another example

What does this do?
```python
A = np.arange(3)   # [0,1,2]
B = np.arange(1,3) # [1,2]
C = einops.einsum(A, B, "i,j->i j")
```

. . .

Both indices are free indices.  There are no summation indices.
```python	
C = np.empty(shape=(3,2))
for i in range(3):      # First outer loop
    for j in range(2):  # Second outer loop
        total = 0
        total += A[i]*B[j] # No inner loop
    C[i,j] = total
```
This is the **outer product** of the two vectors.

## Example without Free Indices

```python
A = np.arange(3)   # [0,1,2]
B = einops.einsum(A, "i->")
```

. . .

This example has no free indices. The index `i` is a summation index.
```python
# No outer loops
total = 0
for i in range(3):  
    total += A[i]
B = total
```
Hence, we summed all elements of the vector `A` to get the scalar `B`.

## Multi Letter Indices

With the einsum implementation of `einops`, one can use multi-letter indices.
```python
A = np.arange(12).reshape(3,4)
B = np.arange(-4,4).reshape(4,2)
C = einops.einsum(A, B, "row inner,inner col->row col")
```  
With multi-dimensional tensors, we can use multi-letter indices to make the code more readable.

# Exercises

## Exercises

Use `einops` to write a Python function to

1. Compute the inner product of two vectors.
2. Transpose a matrix.
3. Compute the sum of each column of a matrix.
   - Result is a vector with the same number of elements as the number of columns.
4. Compute $\mathbf{A}\mathbf{A}^T$ for a matrix $\mathbf{A}$.
5. Compute the trace of a (square) matrix.
    - The trace is the sum of the diagonal elements of a matrix.
6. Compute the Hadamard product of two matrices.
   - The Hadamard product is the element-wise product of two matrices.
  
## Exercise

Reimplement your version of `SimpleMultiHeadAttention` using `einops` and `einsum`.

See notebook:  `multi_head_attention_exercises.ipynb`.

# Other Einops Functions

## `rearrange`

- `rearrange` let's you reorder the axes of a tensor. Common use case is transposition of axes:
```python
print(einops.rearrange(np.arange(12).reshape(3,4), "h w -> w h"))
```
yields
```
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]
```

## `rearrange` for Composition and Decomposition of Axes

See examples in notebook.

## `reduce`

In `einops.reduce`, if an axis is not specified, it is reduced.
```python
X = np.arange(12).reshape(3, 4).astype(np.float32)
einops.reduce(X, "rows cols -> cols", reduction="mean")
```
yields
```
array([4., 5., 6., 7.], dtype=float32)
<array of shape (4,)>
```

## Keeping the reduced axis

```python
# Keep dimensions 
X = np.arange(12).reshape(3, 4).astype(np.float32)
einops.reduce(X, "rows cols -> rows 1", reduction="mean").shape
# or einops.reduce(X, "rows cols -> rows ()", reduction="mean").shape
```
yields
```
(3,1)
```

## Max pooling with `reduce`

```python
# Max pooling and composition of axes
reduce(images, "b (h h2) (w w2) c -> h (b w) c", "max", h2=2, w2=2)
```

## `repeat`

Introduce a new axis.
```python
X = np.arange(4)
einops.repeat(X, "w -> h w", h = 4)
```
yields
```
[[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
```

## `repeat`: shorter syntax

Shorter syntax for `repeat` and different placement of new axis:
```python
X = np.arange(4)
einops.repeat(X, "w -> w 3")
```
yields
```
[[0 0 0]
 [1 1 1]
 [2 2 2]
 [3 3 3]]
```

# Exercises

## Exercises

1. Create the following tensor using only `np.arange` and `einops.rearrange`: 
  ```
  [[3, 4],
   [5, 6],
   [7, 8]]
  ```
2. Create the following tensor using only `np.arange` and `einops.rearrange`:
  ```
  [[1, 2, 3],
  [4, 5, 6]]
  ```
3. Create the following tensor using only `np.arange` and `einops.rearrange`:
  ```
  [[[1], [2], [3], [4], [5], [6]]]
  ```

## Exercises

4. Compute the average temperature for each week given a 1D tensor of temperatures.
  The length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc. Use only `einops` operations.
5. (Continued) For each day, subtract the average for the week the day belongs to. Use only `einops` operations.
6. Normalize the temperatures as follows: for each day, subtract the weekly average and divide by the weekly standard deviation. Use `einops` operations. Pass `np.std` to `reduce`.
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
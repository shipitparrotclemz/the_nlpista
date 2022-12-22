# Comparing scipy.sparse.csr_matrix and numpy.ndarray

A CSR (Compressed Sparse Row) matrix is a sparse matrix.

It has a large number of zero elements. 

In a CSR matrix, the zero elements are not stored in the matrix, which makes it much smaller in size compared to a dense matrix (such as a NumPy array) where all elements, including the zero elements, are stored.

In a CSR matrix, only the non-zero elements are stored, along with their row indices and column indices. 

This representation is more space-efficient than a dense matrix because it avoids storing a large number of zero elements.

For example, consider a dense matrix with the following elements:

```
[ [1, 0, 3],
  [0, 5, 0],
  [2, 0, 4] ]
```

In this matrix, 4 out of 9 elements are non-zero. If we represent this matrix using a CSR matrix, it would look like this:

```
values = [1, 3, 5, 2, 4]
row_indices = [0, 0, 1, 2, 2]
col_indices = [0, 2, 1, 0, 2]
```

The values array stores the non-zero elements

the row_indices array stores the row indices of the non-zero elements

and the col_indices array stores the column indices of the non-zero elements.

Using a CSR matrix can be more efficient in terms of both storage space and computation time when working with sparse matrices

It avoids the overhead of storing and processing a large number of zero elements.

## Experiment - Comparing the memory usage between numpy.nd.array and scipy.sparse.csr_matrix

We compare the memory usage between numpy.nd.array and scipy.sparse.csr_matrix!

We initialize a numpy array of 10,000 rows and 10,000 columns, with an array of 0 elements.

We then create a csr_matrix from the numpy array of the same values and dimensions

And compare the memory differences between the two.

### Results:

```
# Memory usage of X, numpy (MB): 800.000128
# Memory usage of X (MB): 4.8e-05
```

We can see that for a matrix of zero numbers, the CSR matrix uses way less memory compared to a numpy array.

The size of the compressed sparse row matrix will stay the same, as long as the number of non-zero elements stay the same.

In the meantime, the numpy array's size will scale quadratically with the rows / columns of the array.

# Matrix Multiplication between scipy.sparse.csr_matrix and numpy.ndarray

UNDER CONSTRUCTION - `csr_matrix_multiplication.py`

## BUG: numpy (50, 50) dot csr (50, 1), will unexpectedly return (50, 50), not (50, 1)

Source: scipy documentation
- https://docs.scipy.org/doc/scipy/reference/sparse.html#matrix-vector-product

As of NumPy 1.7, np.dot is not aware of sparse matrices, therefore using it will result on unexpected results or errors. The corresponding dense array should be obtained first instead:

```
np.dot(A.toarray(), v)
array([ 1, -3, -1], dtype=int64)
```

but then all the performance advantages would be lost.

## Sources:

Dot product between 1D numpy array and scipy sparse matrix
- https://stackoverflow.com/questions/31040188/dot-product-between-1d-numpy-array-and-scipy-sparse-matrix 
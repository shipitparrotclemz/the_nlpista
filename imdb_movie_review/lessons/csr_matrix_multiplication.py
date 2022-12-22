from datetime import datetime
from typing import Any, Union, Callable

import numpy as np
from scipy.sparse import csr_matrix
import sys


def timeit(func: Callable[[...], Any]) -> Callable[[...], Any]:
    def timed(*args, **kwargs) -> Any:
        start = datetime.utcnow()
        result = func(*args, **kwargs)
        end = datetime.utcnow()
        print(
            f"Time elapsed for {func.__name__}: {end.microsecond - start.microsecond} microsecond"
        )
        return result

    return timed


@timeit
def timed_numpy_dot(
    X: Union[np.ndarray, csr_matrix, list[Any]],
    Y: Union[np.ndarray, csr_matrix, list[Any]],
) -> np.ndarray:
    return X.dot(Y)


@timeit
def timed_at_operator(
    X: Union[np.ndarray, csr_matrix, list[Any]],
    Y: Union[np.ndarray, csr_matrix, list[Any]],
) -> np.ndarray:
    return X @ Y


if __name__ == "__main__":
    """
    Objective:

    Test if both @ and np.dot supports matrix multiplication between
    - numpy.ndarray and numpy.ndarray
    - numpy.ndarray and scipy.sparse.csr_matrix
    - numpy.ndarray and list[list[float]]

    Compare the memory usage differences between
    - using the standard python `@` operator that calls __matmul__, and the numpy.dot method
    """
    # Create a sparse matrix with 50 rows and 50 columns

    X_numpy: np.ndarray = np.zeros((50, 50))

    X_csr: csr_matrix = csr_matrix(X_numpy)

    # Create a NumPy array with 50 rows and 1 column
    W_numpy: np.ndarray = np.random.rand(50, 1)

    # Create a csr matrix of the same dimensions and values
    W_csr: csr_matrix = csr_matrix(W_numpy)

    # Create a 2D list of the same dimensions and values
    W_list: list[list[float]] = [numpy_list.tolist() for numpy_list in W_numpy]

    # Use the @ operator to perform matrix multiplication between the csr matrix and the numpy array
    # the csr matrix gets converted to a numpy array before the matrix multiplication
    Y_csr_numpy: np.ndarray = timed_at_operator(X_csr, W_numpy)
    # Use the dot() function to perform matrix multiplication between the two numpy arrays
    Y_numpy_numpy: np.ndarray = timed_at_operator(X_numpy, W_numpy)
    # Use the dot() function to perform matrix multiplication between the numpy array and list
    Y_numpy_list: np.ndarray = timed_at_operator(X_numpy, W_list)

    # Get the memory usage of the result
    # Memory usage of Y_csr_numpy (MB): 0.000128
    print(f"Y_csr_numpy.shape: {Y_csr_numpy.shape}")
    print("Memory usage of Y_csr_numpy (MB):", sys.getsizeof(Y_csr_numpy) / 1e6)
    # Memory usage of Y_numpy_numpy (MB): 0.000528
    print(f"Y_numpy_numpy.shape: {Y_numpy_numpy.shape}")
    print("Memory usage of Y_numpy_numpy (MB):", sys.getsizeof(Y_numpy_numpy) / 1e6)
    # Memory usage of Y_numpy_list (MB): 0.000528
    print(f"Y_numpy_list.shape: {Y_numpy_list.shape}")
    print("Memory usage of Y_numpy_list (MB):", sys.getsizeof(Y_numpy_list) / 1e6)

    assert (
        Y_csr_numpy.shape == Y_numpy_numpy.shape == Y_numpy_list.shape
    ), "Output shapes are unexpectedly different"

    # Use the dot() function to perform matrix multiplication between the csr matrix and the numpy array
    # the csr matrix gets converted to a numpy array before the matrix multiplication

    """    
    # TODO: BUG: numpy (50, 50) dot csr (50, 1), will unexpectedly return (50, 50), not (50, 1)

    Source: scipy documentation
    - https://docs.scipy.org/doc/scipy/reference/sparse.html#matrix-vector-product
    As of NumPy 1.7, np.dot is not aware of sparse matrices, therefore using it will result on unexpected results or errors. The corresponding dense array should be obtained first instead:
    
    np.dot(A.toarray(), v)
    array([ 1, -3, -1], dtype=int64)
    but then all the performance advantages would be lost.
    """

    # csr (50, 50) dot csr (50, 1) returns (50, 1) as expected
    # numpy (50, 50) dot numpy (50, 1) returns (50, 1) as expected
    Z_csr_numpy: np.ndarray = X_csr.dot(W_numpy)
    # Use the dot() function to perform matrix multiplication between the two numpy arrays
    Z_numpy_numpy: np.ndarray = timed_numpy_dot(X_numpy, W_numpy)
    # Use the dot() function to perform matrix multiplication between the numpy array and list
    Z_numpy_list: np.ndarray = timed_numpy_dot(X_numpy, W_list)

    # Get the memory usage of the result
    # Memory usage of Z_csr_numpy (MB): 0.020128
    print(f"Z_csr_numpy.shape: {Z_csr_numpy.shape}")
    print("Memory usage of Z_csr_numpy (MB):", sys.getsizeof(Z_csr_numpy) / 1e6)
    # Memory usage of Z_numpy_numpy (MB): 0.000528
    print(f"Z_numpy_numpy.shape: {Z_numpy_numpy.shape}")
    print("Memory usage of Z_numpy_numpy (MB):", sys.getsizeof(Z_numpy_numpy) / 1e6)
    # Memory usage of Z_numpy_list (MB): 0.000528
    print(f"Z_numpy_list.shape: {Z_numpy_list.shape}")
    print("Memory usage of Z_numpy_list (MB):", sys.getsizeof(Z_numpy_list) / 1e6)

    assert np.array_equal(
        Y_numpy_numpy, Z_numpy_numpy
    ), "numpy numpy matrix multiplication matrix is unexpectedly different in shape and size"
    assert np.array_equal(
        Y_numpy_list, Z_numpy_list
    ), "numpy list matrix multiplication matrix is unexpectedly different in shape and size"

    # This is not the same!
    # Q: Why does numpy_array.dot(csr_matrix) give a different result than numpy_array @ csr_matrix?
    assert np.array_equal(
        Y_csr_numpy, Z_csr_numpy
    ), "csr numpy matrix multiplication matrix is unexpectedly different in shape and size"

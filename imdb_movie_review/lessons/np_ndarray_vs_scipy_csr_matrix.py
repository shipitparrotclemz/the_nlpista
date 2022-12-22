import sys

import numpy as np
from scipy.sparse import csr_matrix

if __name__ == "__main__":
    """
    Objective: 
    Compare the memory usage differences between
    - scipy.sparse.csr_matrix and np.nd.narray
    
    Answer: scipy.sparse.csr_matrix is way more space efficient than np.nd.array.
    
    The larger the sparse np.ndarray, the more space efficient csr_matrix is compared to the sparse np.ndarray
    """
    X_numpy: np.ndarray = np.random.rand(10_000, 10_000)

    X: csr_matrix = csr_matrix(X_numpy)

    # Get the memory usage of the numpy matrix of X
    # Memory usage of X, numpy (MB): 800.000128
    print("Memory usage of X, numpy (MB):", sys.getsizeof(X_numpy) / 1e6)

    # Get the memory usage of the sparse matrix
    # Memory usage of X (MB): 4.8e-05
    print("Memory usage of X (MB):", sys.getsizeof(X) / 1e6)
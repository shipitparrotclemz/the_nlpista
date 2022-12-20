import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import psutil


def get_memory_usage(message: str) -> None:
    """
    Loading the data into memory, and the training process (fit) can take alot of memory.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    print(message)
    print(f"Total memory used by process: {memory_info.rss / 1024**2:.2f} MB")


if __name__ == "__main__":
    # we load only 10000 out of 50000 rows, due to memory constraints
    # training with bag of words vectorization is very expensive in memory
    # we will cover techniques to reduce memory usage next time.
    rows: int = 10000

    df: pd.DataFrame = pd.read_csv("imdb_dataset.csv")[:rows]

    """
    memory usage after loading 10000 rows of data into memory
    Total memory used by process: 209.77 MB
    """
    get_memory_usage(
        message=f"memory usage after loading {rows} rows of data into memory"
    )

    """
    The review is the feature and the sentiment is what we are trying to predict.
    """
    X: pd.Series = df["review"]
    y: pd.Series = df["sentiment"]

    """
    Vectorize the feature with bag of words
    """
    cv: CountVectorizer = CountVectorizer()

    bag_of_words_X: scipy.sparse._csr.csr_matrix = cv.fit_transform(X)

    """
    memory usage after vectorizing 10000 rows of data with bag of words
    Total memory used by process: 260.53 MB
    """
    get_memory_usage(
        message=f"memory usage after vectorizing {rows} rows of data with bag of words"
    )

    """
    PS: bag of words vectors can be huge and take up alot of memory:
    y.shape: (10000,)
    """
    print(f"bag_of_words_X.shape: {bag_of_words_X.shape}")

    """
    Convert the sentiment to numbers; positive = 1, negative = 0
    """
    y = y.replace("positive", 1).replace("negative", 0)

    print(f"y.shape: {y.shape}")

    """
    Split X and y into train and test sets

    Q: What is a good ratio of training to test sets?
    - Typically, we choose 0.8 to training, and 0.2 to test sets
    - But, this depends on the size of the dataset. If the dataset is relatively small, like this, at 50,000
    We typically increase the training data set ratio, in a bid to have enough data for training.
    something slightly larger at 0.9 to 0.1

    Lets do that with sklearn's train_test_split
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    """
    memory usage before converting bag of words csr matrix to numpy array
    Total memory used by process: 253.28 MB
    """
    get_memory_usage(
        message=f"memory usage before converting bag of words csr matrix to numpy array"
    )

    bag_of_words_X_np: np.ndarray = bag_of_words_X.toarray()

    """
    MOST MEMORY EXPENSIVE OPERATION:
    
    memory usage after converting bag of words csr matrix to numpy array
    Total memory used by process: 2435.41 MB
    """
    get_memory_usage(
        message=f"memory usage after converting bag of words csr matrix to numpy array"
    )

    y_np: np.ndarray = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        bag_of_words_X_np, y_np, test_size=0.1, random_state=42
    )

    """
    memory usage after train test split
    Total memory used by process: 3704.88 MB
    """
    get_memory_usage(message=f"memory usage after train test split")

    """
    Lets use scikitlearn's LogisticRegression
    """

    """
    memory usage before training
    Total memory used by process: 3705.22 MB
    """
    get_memory_usage(message=f"memory usage before training")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    """
    memory usage after training
    Total memory used by process: 3668.78 MB
    """
    get_memory_usage(message=f"memory usage after training")

    """
    Now, we have the next question:

    Q: How do we quantify the performance of our model?

    The most naive way is with Accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    print(f"Accuracy: {model.score(X_test, y_test)}")

    """
    However, Accuracy can be misleading; 

    if we happen to have imbalance of classes; 

    e.g 80% positive reviews, 20% negative reviews in our validation set, 

    even a bad classifier that always returns positive will have a good accuracy of 80%

    Lets quickly check if we have this imbalance of class in our test set
    """
    number_of_positive_reviews: int = y_test == 1
    number_of_negative_reviews: int = y_test == 0

    print(f"total number of reviews in test: {len(y_test)}")
    print(f"number_of_positive_reviews in test : {np.sum(number_of_positive_reviews)}")
    print(f"number_of_negative_reviews in test: {np.sum(number_of_negative_reviews)}")

    get_memory_usage(message="memory usage at the end of script.")

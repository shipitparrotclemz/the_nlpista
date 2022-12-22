import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from our_model_implementations.logistic_regression_implementation import OurLogisticRegression
from utils.memory_utils import get_memory_usage

if __name__ == "__main__":
    rows: int = 50000

    df: pd.DataFrame = pd.read_csv("imdb_dataset.csv")[:rows]

    """
    memory usage after loading 50000 rows of data into memory: Total memory used by process: 217.81 MB
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
    memory usage after vectorizing 50000 rows of data with bag of words: Total memory used by process: 409.50 MB
    """
    get_memory_usage(
        message=f"memory usage after vectorizing {rows} rows of data with bag of words"
    )

    """
    PS: bag of words vectors can be huge and take up alot of memory:
    y.shape: (50000,)
    
    Tip: avoid converting X_train / X_test to a np array!
    
    The converted np array is dramatically larger in memory than the sparse matrix.
    
    MOST MEMORY EXPENSIVE OPERATION:
    
    For 10,000 rows...
    
    ```
    memory usage after converting bag of words csr matrix to numpy array: Total memory used by process: 2435.41 MB
    get_memory_usage(
        message=f"memory usage after converting bag of words csr matrix to numpy array"
    )
    ```
    
    the csr_matrix representation is designed to efficiently store and manipulate sparse matrices, 
    which have a large number of elements that are zero or otherwise absent. 
    
    When we convert a csr_matrix to an ndarray, the resulting array must store the full set of values, 
    including the zero or absent elements, which can significantly increase the required memory.
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

    X_train: scipy.sparse._csr.csr_matrix
    X_test: scipy.sparse._csr.csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray

    y_np: np.ndarray = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        bag_of_words_X, y_np, test_size=0.1, random_state=42
    )

    """
    memory usage after train test split: Total memory used by process: 398.56 MB
    """
    get_memory_usage(message=f"memory usage after train test split")

    """
    Lets use scikitlearn's LogisticRegression
    
    memory usage before training: Total memory used by process: 398.56 MB
    """
    get_memory_usage(message=f"memory usage before training")
    model = OurLogisticRegression(max_iter=20000)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    """
    memory usage after training: Total memory used by process: 490.64 MB
    """
    get_memory_usage(message=f"memory usage after training")

    """
    Now, we have the next question:

    Q: How do we quantify the performance of our model?

    The most naive way is with Accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    # 10,000 iterations - Bag of Words + sklearn Logistic Regression
    # Accuracy: 0.8908
    # 10,000 iterations - Bag of Words + our own Logistic Regression
    # Accuracy: 0.8618
    # 16000 iterations - Bag of Words + our own Logistic Regression
    # Accuracy - 0.8748
    # 20000 iterations - Bag of Words + our own Logistic Regression
    # Accuracy - 0.8738
    """
    Question:
    - Why does scikitlearn's LogisticRegression converge its loss function faster than our implementation?
    """
    print(f"Accuracy: {accuracy_score(X_test, y_test)}")

    """
    However, Accuracy can be misleading; 

    if we happen to have imbalance of classes; 

    e.g 80% positive reviews, 20% negative reviews in our validation set, 

    even a bad classifier that always returns positive will have a good accuracy of 80%

    Lets quickly check if we have this imbalance of class in our test set
    """
    number_of_positive_reviews: int = y_test == 1
    number_of_negative_reviews: int = y_test == 0

    # total number of reviews in test: 5000
    print(f"total number of reviews in test: {len(y_test)}")

    # number_of_positive_reviews in test : 2519
    print(f"number_of_positive_reviews in test : {np.sum(number_of_positive_reviews)}")

    # number_of_negative_reviews in test: 2481
    print(f"number_of_negative_reviews in test: {np.sum(number_of_negative_reviews)}")

    """
    In this case, we do not have a significant imbalance of positive / negative classes in the test set
    
    Lets explore three other metrics, which are robust towards imbalance of positive / negative classes
    
    Precision
    - TP / (TP + FP)
    
    Recall 
    - TP / (TP + FN)
    
    F1 Score
    - 2 * (Precision * Recall) / (Precision + Recall)
    """

    y_pred: np.ndarray = model.predict(X_test)

    # 10,000 iterations - Bag of Words + sklearn Logistic Regression
    # Precision: 0.8855021492770614
    # 10,000 iterations - Bag of Words + our own Logistic Regression
    # Precision: 0.8542635658914729
    # 16000 iterations - Bag of Words + our own Logistic Regression
    # Precision: 0.8675728155339806
    # 20000 iterations - Bag of Words + our own Logistic Regression
    # Precision: 0.8630769230769231
    print(f"Precision: {precision_score(y_test, y_pred)}")

    # 10,000 iterations - Bag of Words + sklearn Logistic Regression
    # Recall: 0.8995633187772926
    # 10,000 iterations - Bag of Words + our own Logistic Regression
    # Recall: 0.8749503771337832
    # 16000 iterations - Bag of Words + our own Logistic Regression
    # Recall: 0.8868598650258039
    # 20000 iterations - Bag of Words + our own Logistic Regression
    # Recall: 0.8908296943231441
    print(f"Recall: {recall_score(y_test, y_pred)}")

    # 10,000 iterations - Bag of Words + sklearn Logistic Regression
    # F1 Score: 0.8924773532886965
    # 10,000 iterations - Bag of Words + our own Logistic Regression
    # F1 Score: 0.8644832320062756
    # 16000 iterations - Bag of Words + our own Logistic Regression
    # F1 Score: 0.8771103258735767
    # 20000 iterations - Bag of Words + our own Logistic Regression
    # F1 Score: 0.876733737058019
    print(f"F1 Score: {f1_score(y_test, y_pred)}")

    """
    The metrics seem to point to the model doing well!
    
    Q: How do we improve this further?
    
    Lets explore this in the next one!
    """

import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv("imdb_dataset.csv")

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
    PS: bag of words can be huge and take up alot of memory:
    (50000, 101895)
    """
    print(f"bag_of_words_X.shape: {bag_of_words_X.shape}")

    """
    Convert the sentiment to numbers; positive = 1, negative = 0
    """
    y = y.replace("positive", 1).replace("negative", 0)

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

    bag_of_words_X_np: np.ndarray = bag_of_words_X.toarray()
    y_np: np.ndarray = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        bag_of_words_X_np, y_np, test_size=0.1, random_state=42
    )

    """
    Lets use scikitlearn's LogisticRegression
    """

    model = LogisticRegression(max_iter=1)
    model.fit(X_train, y_train)

    """
    Now, we have the next question:

    Q: How do we quantify the performance of our model?

    The most naive way is with Accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    # print(f"Accuracy: {model.score(X_test, y_test)}")

    """
    However, Accuracy can be misleading; 

    if we happen to have imbalance of classes; 

    e.g 80% positive reviews, 20% negative reviews in our validation set, 

    even a bad classifier that always returns positive will have a good accuracy of 80%

    Lets quickly check if we have this imbalance of class in our test set
    """

    # print(f"Number of positive reviews: {y_test[y_test['sentiment'] == 1].shape[0]}")
    # print(f"Number of negative reviews: {y_test[y_test['sentiment'] == 0].shape[0]}")

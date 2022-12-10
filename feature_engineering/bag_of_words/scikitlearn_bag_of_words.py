from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse._csr import csr_matrix

"""
In scikit-learn, the CountVectorizer class is a commonly used implementation of bag of words. 

It takes in a list of strings (e.g. a list of emails), and returns a matrix of the counts of each word in the vocabulary.

We use the CountVectorizer to create a bag of words representation of a list of sentences:

The resulting bag of words vectors are represented as a sparse matrix, 

where each row corresponds to an sentence,

and each column corresponds to a word in the vocabulary. 

The entries in the matrix represent the counts of each word in the vocabulary for each email.
"""

if __name__ == "__main__":
    # Create a list of sentences
    sentences: list[str] = [
        "I love sunflower seeds",
        "I hate millet seeds",
    ]

    # Create a CountVectorizer object
    vectorizer: CountVectorizer = CountVectorizer()

    # Fit the vectorizer on the emails, and transform them into bag of words vectors
    # X is a compressed, sparse row matrix
    X: csr_matrix = vectorizer.fit_transform(sentences)

    # Print the feature words
    # feature names: {'love': 1, 'sunflower': 4, 'seeds': 3, 'hate': 0, 'millet': 2}
    print(f"feature names: {vectorizer.vocabulary_}")

    # Print the bag of words vectors
    print(f"matrix: {X}")
    """
    matrix:   
    1st sentence: "I love sunflower seeds"
    2nd sentence: "I hate millet seeds",
    feature names: {'love': 1, 'sunflower': 4, 'seeds': 3, 'hate': 0, 'millet': 2}
    
    matrix:      
    word           count
    
    1st sentence: "I love sunflower seeds"
    (0, 1)        1     # love
    (0, 4)        1     # sunflower
    (0, 3)        1     # seeds
    
    2nd sentence: "I hate millet seeds",
    (1, 3)        1     # seeds
    (1, 0)        1     # hate
    (1, 2)        1     # millet

    """

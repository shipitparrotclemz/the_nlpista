import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    """
    Can we predict if a candy is chocolate or not based on its other features?
    """

    data: pd.DataFrame = pd.read_csv("candy-data.csv")

    """
    Separate the data columns into two;
    Y is the chocolate column, the column we want to predict
    X are the columns we will be using to predict Y
    """
    X: pd.DataFrame
    y: pd.DataFrame
    X, Y = data.drop(["chocolate", "competitorname"], axis=1), data[["chocolate"]]

    """
    Print the first 5 rows of X
    
       fruity  caramel  peanutyalmondy  nougat  crispedricewafer  hard  bar  pluribus  sugarpercent  pricepercent  winpercent
    0       0        1               0       0                 1     0    1         0         0.732         0.860   66.971725
    1       0        0               0       1                 0     0    1         0         0.604         0.511   67.602936
    2       0        0               0       0                 0     0    0         0         0.011         0.116   32.261086
    3       0        0               0       0                 0     0    0         0         0.011         0.511   46.116505
    4       1        0               0       0                 0     0    0         0         0.906         0.511   52.341465

    """
    print(X.head())

    """
    Print the first 5 rows of Y
    
       chocolate
    0          1
    1          1
    2          0
    3          0
    4          0
    """
    print(Y.head())

    """
    Now, lets split the data into two sets; 
    X_train and y_train are the data we will use to train our model
    X_test and y_test are the data we will use to test our model
    
    Lets use the train_test_split function from sklearn to split the data
    
    Question:
    - What is a good ratio of training data to test data?
    
    Answer:
    - There is no one "correct" ratio of training data to test data for any other machine learning model. 
    
    - The appropriate ratio depends on the size and quality of your dataset, goals and requirements of your project.

    - As a general rule, it is recommended to reserve a portion of your data for testing, 
    to ensure that the model is able to generalize well to new data. 
    - A common approach is to use a split of 80% training data and 20% test data.
    
    - It's also worth noting that if you have a very small dataset, like what we have in this case
    it may not be possible to split it into a training and test set without losing too much data for training. 
        
    - Ultimately, the most important thing is to ensure that your model has been trained and evaluated on a diverse 
    and representative sample of the data it will be applied to, 
    so that you can be confident in its ability to generalize to new data.
    
    Since we have relatively few, 85 rows of data, 
    - we can increase the training data and decrease the testing data to 90% training, 10% testing
    
    This gives us more data to train the model on.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    """
    Lets train a LogisticRegression classifier
    
    Question:
    - How do we know if we should stop training the classifier?
    
    Answer:
    - A good guess can be, the classifier performance not improving much after a certain number of iterations.
    
    and this gives us more questions!
    
    Question: How do we measure performance of the model during training?
    
    Question: How do we set a good threshold in the change of performance after each iteration of training?
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)

    """
    Now, lets test the model on the test data, by measuring its accuracy!
    
    TP - True Positive
    True positives are cases where the model correctly predicts the candy is a chocolate candy
    
    TN - True Negative
    True negatives are cases where the model correctly predicts the candy is not a chocolate candy

    FP - False Positive
    False positives are cases where the model incorrectly predicts the candy is a chocolate candy, but its really not.
    
    FN - False Negative
    False positives are cases where the model incorrectly predicts the candy is not a chocolate candy, but it really is one.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    accuracy: float = model.score(X_test, y_test)

    """
    model_score: 0.8888888888888888
    mean accuracy on the given test data and labels.
    """
    print(f"accuracy: {accuracy}")

    """
    Accuracy is not a great metric for the performance of the model, 
    
    if there is a significant difference in the number of observations for each class
    
    For example, for spam detection, most emails in inboxes might be spam.
    
    For example, lets say 90% of the dataset contains spam
    
    If we have a terribly bad classifier which always predicts an email as spam.
    
    It would have a misleading accuracy of 90%! The classifier sucks! But we have an accuracy of 90% for it!
    
    Q: What is a better metric?
    
    1. Precision: Proportion of True Positives among all Positives by the Model
    
    TP / (TP + FP) 
    
    2. Recall: Proportion of True Positives among all Actual Positives
    
    TP / (TP + FN) 
            
    3. F1 Score: Weighted Average of Precision and Recall
    
    Recall = TP / (TP + FN)
    
    The F1 score is a balance between precision and recall, and it is often used when there is an uneven class distribution, 
    or when it is important to avoid false positives or false negatives. F
    
    or example, in a medical diagnosis task, 
    it might be more important to avoid false negatives 
    (predicting that a patient does not have a disease when they actually do) 
    than false positives (predicting that a patient has a disease when they actually do not). 
    
    In this case, the F1 score would be a good metric to use, as it takes both precision and recall into account.
    """

    precision: float = precision_score(y_test, model.predict(X_test))
    recall: float = recall_score(y_test, model.predict(X_test))
    f1_score: float = f1_score(y_test, model.predict(X_test))

    # precision: 1.0
    print(f"precision: {precision}")
    # recall: 0.75
    print(f"recall: {recall}")
    # f1_score: 0.8571428571428571
    print(f"f1_score: {f1_score}")

    """
    Pretty good! but...
    
    Question: Is there a way to further improve the recall and f1 score?
    """
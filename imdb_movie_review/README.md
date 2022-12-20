# Classifying the sentiment of IMDB movie reviews with Bag of Words and Logistic Regression

In this demonstration, we will be training our own model to classify a movie review as a good or a bad movie!

## Feature Engineering - Bag of Words

As models don't understand text as it is, and only understand numbers...

we will be feature engineering movie reviews into input vectors with bag of words.

## Splitting the data for training and testing

We have two objectives; 

1. We want a reliable model which can classify movie reviews correctly as positive or negative. 

This means, we need enough training data which are representative of movie reviews in the real world, and work well on movie reviews it has never seen before.

2. We want a reliable test of our trained model, with enough data which are not in the training dataset.

This means, we need enough data to give a good representative test for the trained model's performance.

The data should also not be in the training data set, for a fair test.

Q: What is a good ratio for train and test?

We will be exploring how we can split the data easily with `sklearn.model_selection.train_test_split`.

## Training the model

We will be using logistic regression to classify the sentiment of the reviews.

We will attempt implementing our own Logistic Regression in a step-by-step fashion.

we will also explore using a production-ready `sklearn.linear_model.LogisticRegression` class.

## Validating the performance of the model

We will explore using Accuracy as a performance metric, and how it can be misleading if we have an imbalanced dataset of positive and negative reviews.

We will also explore using Precision and Recall as performance metrics, and how they can be used to evaluate the performance of our model.

Finally, we will also explore F1-Score as a performance metric, and how it weights the model based on a balance of Precision and Recall.

## Dataset Source

IMDB Dataset of 50K Movie Reviews
- https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/code
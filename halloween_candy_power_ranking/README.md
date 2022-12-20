# Logistic Regression with Halloween Candy Data

## Content

`candy-data.csv` includes attributes for each candy along with its ranking. For binary variables, 1 means yes, 0 means no. The data contains the following fields:

chocolate: Does it contain chocolate?
fruity: Is it fruit flavored?
caramel: Is there caramel in the candy?
peanutalmondy: Does it contain peanuts, peanut butter or almonds?
nougat: Does it contain nougat?
crispedricewafer: Does it contain crisped rice, wafers, or a cookie component?
hard: Is it a hard candy?
bar: Is it a candy bar?
pluribus: Is it one of many candies in a bag or box?
sugarpercent: The percentile of sugar it falls under within the data set.
pricepercent: The unit price percentile compared to the rest of the set.
winpercent: The overall win percentage according to 269,000 matchups.

## Creating the virtual environment and installing the requirements

```commandline
virtualenv venv -p $(which python3.9)
source venv/bin/activate
pip install -r requirements.txt
```

## Understanding the data:

Follow us, as we explore the data in `candy-data.csv!`

1. Use pandas to read in the csv file with the `pandas.read_csv` method, into an efficient, tabular `pandas.DataFrame` data structure.
2. Get the first 5 rows of the data frame with the `.head()` method
3. Get the columns with `.columns` attribute on the data frame
4. Get the data types of each column with the `.dtypes` attribute on the data frame

```commandline
python3 exploratory_data_analysis.py
```

## Building a chocolate candy classifier:

```commandline
python3 predicting_chocolate.py
```

### Doing a train-test split on the training data;

1. Use the `train_test_split` method from `sklearn.model_selection` to split the data into training and testing sets.

### Training a Logistic Regression Model to predict if a candy has chocolate or not.

1. Use the `LogisticRegression` class from `sklearn.linear_model` to create a logistic regression model.

### Evaluating the model with Accuracy, Precision, Recall and F1-Score

1. Use the `accuracy_score`, `precision_score`, `recall_score` and `f1_score` methods from `sklearn.metrics` to evaluate the model.

## Data Source:
- https://www.kaggle.com/datasets/fivethirtyeight/the-ultimate-halloween-candy-power-ranking

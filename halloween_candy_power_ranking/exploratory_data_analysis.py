import pandas
import pandas as pd

if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv("candy-data.csv")

    """
    Print the first 5 rows in the dataset

      competitorname  chocolate  fruity  caramel  peanutyalmondy  nougat  crispedricewafer  hard  bar  pluribus  sugarpercent  pricepercent  winpercent
    0      100 Grand          1       0        1               0       0                 1     0    1         0         0.732         0.860   66.971725
    1   3 Musketeers          1       0        0               0       1                 0     0    1         0         0.604         0.511   67.602936
    2       One dime          0       0        0               0       0                 0     0    0         0         0.011         0.116   32.261086
    3    One quarter          0       0        0               0       0                 0     0    0         0         0.011         0.511   46.116505
    4      Air Heads          0       1        0               0       0                 0     0    0         0         0.906         0.511   52.341465
    """
    print(df.head())

    """
    1. How many candies are in the dataset?
    There are 85 candies in the dataset.
    """
    print(f"There are {len(df)} candies in the dataset.")

    """
    2. How many columns are in the dataset?
    There are 13 columns in the dataset.
    """
    print(f"There are {len(df.columns)} columns in the dataset.")

    """
    The column names are Index(['competitorname', 'chocolate', 'fruity', 'caramel', 'peanutyalmondy',
       'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent',
       'pricepercent', 'winpercent'],
       
    3. What are the column names?
    """
    print(f"The column names are {df.columns}.")

    """
    # 4. What are the data types of the columns?

    The data types of the columns are 
    competitorname       object
    chocolate             int64
    fruity                int64
    caramel               int64
    peanutyalmondy        int64
    nougat                int64
    crispedricewafer      int64
    hard                  int64
    bar                   int64
    pluribus              int64
    sugarpercent        float64
    pricepercent        float64
    winpercent          float64
    """
    print(f"The data types of the columns are {df.dtypes}.")
import pandas as pd

if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv("imdb_dataset.csv")

    """
    Explore the data by printing the first five rows in the dataset
    
                                                  review sentiment
    0  One of the other reviewers has mentioned that ...  positive
    1  A wonderful little production. <br /><br />The...  positive
    2  I thought this was a wonderful way to spend ti...  positive
    3  Basically there's a family where a little boy ...  negative
    4  Petter Mattei's "Love in the Time of Money" is...  positive
    """
    print(df.head())

    """
    Get the number of rows and columns in the dataset by printing shape the shape 
    rows: 50000
    columns: 2
    """
    rows: int
    cols: int
    rows, cols = df.shape
    print(f"rows: {rows}")
    print(f"columns: {cols}")

    """
    Get the column names of the dataset
    column names: Index(['review', 'sentiment'], dtype='object')
    """
    column_names: pd.Index = df.columns
    print(f"column names: {column_names}")

    """
    Get the data types of each column name
    data types: review       object
    sentiment    object
    dtype: object
    """
    data_types: pd.Series = df.dtypes
    print(f"data types: {data_types}")
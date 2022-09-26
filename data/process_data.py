"""
PREPROCESSING DATA
Disaster Response Pipeline Project
Udacity - Data Science Nanodegree

Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Arguments:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)
"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Data function

    Arguments:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    Output:
        df -> Loaded dasa as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """
    Clean Data function

    Arguments:
        df -> raw data Pandas DataFrame
    Outputs:
        df -> clean data Pandas DataFrame
    """
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0,]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(
            lambda x: x.split("-")[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # df = pd.concat([df,pd.get_dummies(df.genre)],axis=1)
    # df = df.drop(['genre','social'],axis=1)
    # drop duplicates
    df = df.drop_duplicates(subset=['message'])
    return df


def save_data(df, database_filename):
    """
    Save Data function

    Arguments:
        df -> Clean data Pandas DataFrame
        database_filename -> database file (.db) destination path
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False)
    pass


def main():
    """
    Main Data Processing function

    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database
    """
    print(sys.argv)
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

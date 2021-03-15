# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """Loads and merges messages and categories datasets
    
    In:
    messages_filepath - The filepath of the messages dataset.
    categories_filepath - The filepath of the categories dataset.
       
    Out:
    df dataframe. Dataframe containing merged messages and categories datasets.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath) 
    df = pd.merge(messages, categories , how='left', on=['id'])
    
    return df

    
def clean_data(df):
    """Clean the dataframe - 
    
    IN:
    df - a dataframe we previously created by merging the two datasets
       
    OUT:
    df - a modified and cleaned dataframe
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    # NOTE: I decided not to use the lambda, it was easier for me to do it this way:

    category_colnames = [x[0:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric, errors='coerce')
    # drop the original categories column from `df`
    df = df.drop(['categories'] , axis=1)
   
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df , categories] , axis=1)
    df.head()
    # drop duplicates
    df.drop_duplicates(inplace=True)
   
    return df
        
def save_data(df, database_filename):
    """
    Saving the clean data into a sqlite database
    IN:
    The dataframe that we created earlier
    OUT:
    SQLite DataBase containing the data from the dataframe.
    """
   #commented for a backup
    #engine = create_engine('sqlite:///DisasterResponse.db')
    #df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

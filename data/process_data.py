import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages a the categories from two csv files and merges them to a dataframe.
    
    Args:
        (String) messages_filepath: the filepath for the messages
        (String) categories_filepath: the filepath for the categories
    
    Returns:
        (DataFrame) df: the merged dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, left_on='id', right_on='id', how='outer')


def clean_data(df):
    """
    Cleans the dataframe. Cleaning contains the steps:
    1. Spliting the string in the column categories to build one column for each category
    2. Filling the values for each cell
    3. Drop the old column categories
    
    Args:
        (DataFrame) df: the dataframe to be cleaned
    
    Returns:
        (DataFrame) df: the cleaned dataframe
    """
    
    cat_df = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = cat_df[:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = []
    for column in row.columns:
        colname = row[column].str.split('-')[0][0]
        category_colnames.append(colname)
        
    cat_df.columns = category_colnames

    for column in cat_df:
        # set each value to be the last character of the string
        cat_df[column] = cat_df[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        cat_df[column] = pd.to_numeric(cat_df[column])
        
    df = df.drop(['categories'], axis=1)
 
    df = pd.concat([df, cat_df], axis=1)
    
    df = df.drop_duplicates()
    
    return df
    
def save_data(df, database_filepath):
    """
    Saves the dataframe to sqlite file
    
    Args:
        (DataFrame) df: the dataframe to be saved
        (String) database_filename: the filename of the sqlite file
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Message', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        #messages_filepath = 'messages.csv'
        #categories_filepath = 'categories.csv'
        
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
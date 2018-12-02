# import packages
import sys
import re
import pickle

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Loads message and category data from the surpassed sql_source.
    Args: 
        (String) database_filepath: filename of the sql file
    Returns:  
        (DataFrame) X: feature
        (DataFrame) Y: labels
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Message", engine)

    # select feature data
    X = df.filter(items=['id', 'message', 'original', 'genre'])
    
    # drop feature columns of the dataset: id, message, original, genre
    # drop columns without a message: child_alone
    Y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'],  axis=1).astype(float)
    
    # transform all it on binary classification
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)
    
    return X, Y, df.columns[4:]

def tokenize(text):
    """
    Normalizes, tokenizes, removes stopwords and lemmatizes a given text.
    Args: 
        (String) text: the text to tokenize
    
    Returns:
        (Array) lemmed_tokens: tokenized text
        
    """
    # Normalize text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenizes text
    tokens = word_tokenize(text)
    
    # removes default stopwords
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmed_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return lemmed_tokens 


def build_model():
    """
    Builds a model pipeline.
    
    Returns:
        (GridSearchCV) model_pipeline: the model pipeline
        
    """
    
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 30],
        'clf__estimator__min_samples_split': [2, 4], 
        'vect__ngram_range': [(1, 1), (1, 2)], 
        'clf__estimator__max_depth': [5, 10, 15]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)
    
    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Trains a model pipeline.
    
    Args:
        (GridSearchCV) model: the model pipeline to be tested
        (DataFrame) X_test: Test features
        (DataFrame) Y_test: Test labels
        (Array) category_names: String array of category names
    
    Returns:
        (GridSearchCV) model: the trained model pipeline
    """

    # output model test results
    Y_pred = model.predict(X_test['message'])
    print(classification_report(Y_pred, Y_test, target_names=category_names))



def save_model(model, model_filepath):
    """
    Exports the model pipeline to a python pkl file.
    
    Args:
        (GridSearchCV) model: the model pipeline to be saved
        (String) model_filepath: the filepath of the pkl file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        #database_filepath = 'NewDisasterResponse.db'
        #model_filepath = 'classifier.pkl'
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
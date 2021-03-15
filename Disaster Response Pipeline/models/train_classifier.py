import sys
# import libraries
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

import sqlite3
from sqlalchemy import create_engine

import re

import pickle

def load_data(database_filepath):
    """
    Loads data from database into dataframes
    Input:
    Database file path
    Output:
    X -- ML Input dataframe
    Y -- ML output dataframe with classes as columns
    category_names -- classes names
    
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from DisasterResponsetbl', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    # Remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    tokens = []
    for word in words:
        token = lemmatizer.lemmatize(word).lower().strip()
        tokens.append(token)
        
    return tokens


def build_model(vectorizer='tfidf', classifier='Random Forst'):
    """
    This function splits the data into training and test sets and fit the model into this data
    
    Input:
    vectorizer -- takes one of two inputs 'count' or 'tfidf'
    classifier -- takes one of 3 inputs 'Random Forest', 'Decision Tree' or 'KNN'
    
    Output:
    model fitted
    
    """
    if vectorizer == 'count':
        if classifier == 'Random Forest':
            pipeline = Pipeline([
                ('countvect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('classifier', MultiOutputClassifier(RandomForestClassifier()))
            ])
            
            parameters = {
                'classifier__estimator__n_estimators': (1, 4, 5, 10),
                'classifier__estimator__max_depth': (5, 10)
            }
            
        elif classifier == 'Decision Tree':
            pipeline = Pipeline([
                ('countvect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('classifier', MultiOutputClassifier(DecisionTreeClassifier()))
            ])
            
            parameters = {
                'classifier__estimator__max_depth': [None, 5, 10]
            }
        elif classifier == 'KNN':
            pipeline = Pipeline([
                ('countvect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('classifier', MultiOutputClassifier(KNeighborsClassifier()))
            ])
            
            parameters = {    
                'classifier__estimator__n_jobs': [1, 3, 5],
                'classifier__estimator__n_neighbors': [5, 7, 10]
                
            }
            
    elif vectorizer == 'tfidf':
        if classifier == 'Random Forest':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                ('classifier', MultiOutputClassifier(RandomForestClassifier()))
            ])
            
            parameters = {
                'classifier__estimator__n_estimators': (1, 4, 5, 10),
                'classifier__estimator__max_depth': (5, 10)
            }
            
        elif classifier == 'Decision Tree':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                ('classifier', MultiOutputClassifier(DecisionTreeClassifier()))
            ])
            
            parameters = {
                'classifier__estimator__max_depth': [None, 5, 10]
            }
        elif classifier == 'KNN':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                ('classifier', MultiOutputClassifier(KNeighborsClassifier()))
            ])
            
            parameters = {    
                'classifier__estimator__n_jobs': [1, 3, 5],
                'classifier__estimator__n_neighbors': [5, 7, 10]
                
            }    
    
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    
    

    cv = GridSearchCV(pipeline, param_grid=parameters)


#     cv.fit(X_train, Y_train)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function uses the ML model to predict the output of any new data given and prints the classification result report as Precision, Recall and F1-score
    
    Input:
    model -- ML model or classifier
    X_test -- The testing data
    Y_test -- the correct result of testing data
    category_names -- classes names
    
    Output:
    None
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = category_names
    
    for col in category_names:
        print("For column {}".format(col))
        labels = [x for x in np.unique([Y_test[col] + y_pred[col]]) if str(x) != 'nan']  
        print(classification_report(np.array(Y_test[col]),np.array(y_pred[col]), labels=labels))



def save_model(model, model_filepath):
    """
    save ML model into pickle file
    
    Input:
    model -- ML model or classifier
    model_filepath -- path to save the pickle file

    Output:
    None
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))
 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model('tfidf', 'Random Forest')
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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
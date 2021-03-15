# import libraries
import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle

from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download(['punkt', 'wordnet','stopwords'])


def load_data(database_filepath):
    """
    Load the dataset we created in process_data.py and prepare the variables
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.loc[: , 'related':'direct_report']
    categories = Y.columns.tolist()
    return X, Y, categories

def tokenize(text):
    """
    Tokenize the text
    IN:
    The pre-processed message
    OUT:
    Lemmatized, cleaned and normalized tokens
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    #lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Building a pipeline and using GridSearch to find the best parameteres
    I chose the second model I have previously developed, as it performed slightly better
    IN: 
    The data goes through the designed Pipeline and Grid Search
    OUT:
    After going through Grid Search, the best parameters are chosen and the model is trained
    """
    pipeline2 = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])
    parameters2 = {
    'vect__max_df': (0.5 , 1.0),
    #'clf__estimator__learning_rate': (0.2 , 0.5 , 1.0),
    'clf__estimator__n_estimators': (20, 50, 100)
    }

    cv2 = GridSearchCV(pipeline2, param_grid = parameters2, n_jobs=-2, verbose = 3)
    return cv2

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Testing the model 
    IN:
    The trained model
    OUT:
    Model classification and scores
    """
    Y_pred = model.predict(X_test)
    category_names = Y_test.columns.tolist()
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    for i in range(36):
        print('\n\n  ',category_names[i],':'\
            '\n',\
          classification_report(Y_test.iloc[:,i], Y_pred_df.iloc[:,i]))
    #get best parameters
    best_parameters2 = model.best_params_
    #get best accuracy
    best_accuracy2 = model.best_score_
    # get the average score
    # [commented, currently not in use]average_score_model2 = (Y_pred == Y_test).mean().mean()
    #print the best parameters and best accuracy
    print("Best parameters: ",best_parameters2)
    print("Best score: ",best_accuracy2)
    
def save_model(model, model_filepath):
    """
    Saving the final model 
    IN: 
    The model that has just been built 
    OUT:
    Model file in .sav format
    """
    pickle.dump(model, open('model.sav', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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

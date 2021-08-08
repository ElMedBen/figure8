
import sqlite3
import pandas as pd
import re
import sys
from joblib import dump
import nltk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


#------------------------------------------------------------------------------

def load_data(database_filepath):
    
    #create engine connection to database
    con = sqlite3.connect("{}".format(database_filepath))
    cur = con.cursor()
    
    #read the sql db with a pandas query
    df = pd.read_sql("SELECT * FROM messages", con)
    
    #seperate the needed data and drop unused columns
    X = df["message"].tolist()
    y = df.drop(columns=["message", "index"])
    categories_names = y.columns
    
    #close connection to engine
    con.close()
    
    return X, y, categories_names

#------------------------------------------------------------------------------

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls=re.findall(url_regex,text)

    for url in detected_urls:
        text=text.replace(url, 'urlplaceholder')

    tokens=word_tokenize(text)

    lemmatizer=WordNetLemmatizer()

    clean_tokens=[]

    for tok in tokens : 
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens    

#------------------------------------------------------------------------------

def build_model():

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("multi_out_clf", MultiOutputClassifier(AdaBoostClassifier())),
        ]
    )
    


    return pipeline

#------------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    
    
    y_pred = model.predict(X_test)

    for idv, label in enumerate(y_test.columns):
        print(classification_report(y_test[label], y_pred[:, idv]))
        
#------------------------------------------------------------------------------

def save_model(model, model_filepath):
    dump(model, model_filepath)

#------------------------------------------------------------------------------

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        y = y.drop(columns="genre")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!\n')
        
        print('overall testing score is : ', model.score(X_test,y_test),'\n')
        
        print('overall training score is : ', model.score(X_train,y_train))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py data/DisasterResponse.db models/classifier.pkl')

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
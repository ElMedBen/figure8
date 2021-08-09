# importing all the required libraries

import sqlite3
import pandas as pd
import re
import sys
from joblib import dump
import nltk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ------------------------------------------------------------------------------

# loading data from the database
def load_data(database_filepath):
    """
    Function that loads data from stored database after ETL pipline

    args :
        database_filepath : relative filepath of the database within the project folder

    returns :
        X : table containing the messages preprocessed
        y : the labels that will be predicted based on the text message
        categories_names: names of the categories of messages in a list
    """

    # create engine connection to database
    con = sqlite3.connect("{}".format(database_filepath))
    cur = con.cursor()

    # read the sql db with a pandas query
    df = pd.read_sql("SELECT * FROM messages", con)

    # seperate the needed data and drop unused columns
    X = df["message"].tolist()
    y = df.drop(columns=["message", "index"])
    categories_names = y.columns

    # close connection to engine
    con.close()

    return X, y, categories_names


# ------------------------------------------------------------------------------

# tokenizing data for NLP process
def tokenize(text):
    """
    Function that apply simple preprocessing for nlp by removing url, tokenizing and lematizing. It outputs clean tokens

    args:
        text : text element from the base list that will be preprocessed

    returns:
        clean_tokens: list of clean tokens
    """

    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ------------------------------------------------------------------------------
# building model and preparing the parameters
def build_model():
    """
    Function that build the model by preparing the parameters of gridsearch and instantiating the models
    args:
        None
    returns:
        cv : gridsearch model
    """

    DTC = DecisionTreeClassifier(
        random_state=11, max_features="auto", class_weight="auto", max_depth=None
    )

    parameters = {
        "vect__max_df": [0.5, 1.0],
        "tfidf__use_idf": (True, False),
        "multi_out_clf__estimator__learning_rate": [0.01, 0.1, 1],
        "multi_out_clf__estimator__n_estimators": [1, 10],
    }

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("multi_out_clf", MultiOutputClassifier(AdaBoostClassifier())),
        ]
    )

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2)

    return cv


# ------------------------------------------------------------------------------

# Function to evaluate the model


def evaluate_model(model, X_test, y_test):
    """
    Function that helps to evaluate the model by printing the classification report for each label predicted. It also present the best paramters used in grid search
    args:
        model: built and fited model to training data
        X_test: testing messages selected from the train, test split
        y_test: testing labels selected from the train, test split
    returns:
        printed best parameters from gridsearch and classification report for each label

    """
    model.best_estimator_.steps

    y_pred = model.predict(X_test)
    for idv, label in enumerate(y_test.columns):
        print(label, "\n")
        print(classification_report(y_test[label], y_pred[:, idv]))


# ------------------------------------------------------------------------------

# saving model for later usage
def save_model(model, model_filepath):
    """
    Function that dumps the model as a pkl for later use. Just make sur to put .pkl at the end of the name
    args:
        model: pretrained and fitted model
        model_filepath: file path to save the model with .pkl as format
    returns:
        dumped model as pkl in filepath
    """
    dump(model, model_filepath)


# ------------------------------------------------------------------------------

# main function to execute the full process
def main():
    """
    Function that executes the main process by leveraging all functions
    args:
        None
    returns:
        Execute the full process from loading to printing scores and save the pretrained model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        y = y.drop(columns="genre")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, y_test)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!\n")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "models/train_classifier.py data/DisasterResponse.db models/classifier.pkl"
        )


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

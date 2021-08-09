# importing necessary libraries
import json
import plotly
import pandas as pd
from joblib import load

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# instantiating the flask app
app = Flask(__name__)

# simple tokenizing function for input text
def tokenize(text):
    """
    Function that apply simple preprocessing for nlp by removing url, tokenizing and lematizing. It outputs clean tokens

    args:
        text : text element from the base list that will be preprocessed

    returns:
        clean_tokens: list of clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///data/DisasterResponse.db")
df = pd.read_sql_table("messages", engine)

# load model
model = load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    # genre counts for plotting
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # related stat count for ploting
    related_counts = df.groupby("related").count()["message"]
    related_names = ["not related", "related"]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Bar(x=related_names, y=related_counts)],
            "layout": {
                "title": "Number of messages classified as related or not",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()

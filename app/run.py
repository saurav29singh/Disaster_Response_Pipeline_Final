import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.iloc[:, 4:].columns
    category_boolean = (df.iloc[:, 4:] != 0).sum().values


    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    top_response_counts = Y.sum(axis=0).sort_values(ascending=False)[:10]
    top_response_names = list(top_response_counts.index)

    # create visuals

    graphs = [
        # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_counts,
                    y=genre_names,

                    texttemplate = '%{x}', textposition = 'outside',
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
        # GRAPH 2 - category graph
        {
            'data': [
                Bar(
                    x=top_response_names,
                    y=list(top_response_counts),
                    texttemplate='%{y}', textposition='outside',

                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Message Categories',
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35

                },
                'yaxis': {
                    'title': "Count"


                }
            }
        }
        ,
        # GRAPH 3 - category graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    texttemplate='%{y}', textposition='outside',
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35

                },
                'yaxis': {
                    'title': "Count"

                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
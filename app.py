# You need to run Embeddings before using this codemost_similar = '\n'.join(row[1] for row in most_similar_df.values)

# You probably don't need all these; I just used them all at some point trying things out
import csv
import nltk
import numpy as np
import openai
import pandas as pd
import json
from flask import Flask, render_template, request, url_for, flash, session, redirect, jsonify
import requests
import os

# create the flask app
app = Flask(__name__)
# this is just here because it is needed to maintain session variables, it doesn't matter
app.secret_key = 'BAD_SECRET_KEY'

with open("APIkey.txt") as f:
    openai.api_key = f.read().strip()
    
@app.route('/', methods=('GET', 'POST'))

def index():
    if request.method == 'POST':
       # Load the text and its embeddings
        df_chunks = pd.read_csv('textchunks.csv')

        # create embedding for the query
        embedthequery = openai.Embedding.create(
          model="text-embedding-ada-002",
          input=request.form['content1']
        )

        query_embed=embedthequery["data"][0]["embedding"]

        # function to compute dot product similarity
        def compute_similarity(embedding):
            # convert embedding string to list of floats
            embedding = json.loads(embedding)['embedding']
            embedding = np.array(embedding)
            # compute dot product
            similarity = np.dot(embedding, query_embed)
            return similarity

        # compute similarity for each row and add to new column
        df_chunks['similarity'] = df_chunks['article_embeddings'].apply(lambda x: compute_similarity(x))
        # sort by similarity in descending order
        df_chunks = df_chunks.sort_values(by='similarity', ascending=False)
        # Select the top query_similar_number most similar articles
        most_similar_df = df_chunks.head(8)
        # Get the paper title of the most similar text
        first_row_first_elem = most_similar_df.iloc[0, 0]
        # Concatenate most similar strings, including the first element of the first row at the beginning
        most_similar = "The best guess at a related paper is " + first_row_first_elem + "\n\n" + '\n\n'.join(row[1] for row in most_similar_df.values)

        # I use temperature=0 to give most "factual" answer
        instructions = "Pretend you are a expert graduate student researcher.  You have been asked to answer a series of conceptual questions about some academic papers.  Answer truthfully. Just give the answer in a couple sentences, without language like 'the text above shows' or 'the passage shows' or 'in the text' - I just want the pure response. Say 'I don't know' if you can't find the answer in the text below."
        reply = []
        send_to_gpt = []
        send_to_gpt.append({"role":"system","content":instructions})
        send_to_gpt.append({"role":"assistant","content":most_similar})
        send_to_gpt.append({"role":"user","content":request.form['content1']})
        response=openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=send_to_gpt
        )
        query = request.form['content1']
        tokens_new = response["usage"]["total_tokens"]
        reply = response["choices"][0]["message"]["content"]
        response = jsonify({'reply': reply})
        response.headers['Content-Type'] = 'application/json'
        return response
    else:
        return render_template('index.html')
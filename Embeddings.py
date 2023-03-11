# This code takes all pdfs in documents, scans them, then generates paragraph embeddings on a sliding scale
# Run this before you run app.py
# You need an OpenAI key saved in APIkey.txt
# It is useful to change the file names to a longer title that summarizes briefly what the document is about

import os
from PyPDF2 import PdfReader 
import nltk
import pandas as pd
import openai

# Set the desired chunk size and overlap size
# Make sure all pdfs are in the working directory in a folder called Documents
# chunk_size is how many tokens we will take in each block of text
# overlap_size is how much overlap. So 200, 100 gives you chunks of between the 1st and 200th word, the 100th and 300th, the 200 and 400th...
# I have in no way optimized these
chunk_size = 200
overlap_size = 100
embeddingmodel = "text-embedding-ada-002"
with open("APIkey.txt") as f:
    openai.api_key = f.read().strip()


# Create an empty DataFrame to store the text and title of each document
df = pd.DataFrame(columns=["Title", "Text"])

# Create an empty DataFrame to store the text and title of each document
df = pd.DataFrame(columns=["Title", "Text"])

# Loop through all pdf, txt, tex in the "documents" folder
for filename in os.listdir("Documents"):
    if filename.endswith(".pdf"):
        # Open the PDF file in read-binary mode
        filepath = os.path.join("Documents", filename)
        reader = PdfReader(filepath)
        
        # Extract the text from each page of the PDF
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Add the text and title to the DataFrame
        title = os.path.splitext(filename)[0] # Remove the file extension from the filename
        df = df.append({"Title": title, "Text": text}, ignore_index=True)

    elif filename.endswith(".txt"):
        # Open the text file and read its contents
        filepath = os.path.join("Documents", filename)
        with open(filepath, "r") as file:
            text = file.read()
        
        # Add the text and title to the DataFrame
        title = os.path.splitext(filename)[0] # Remove the file extension from the filename
        df = df.append({"Title": title, "Text": text}, ignore_index=True)
        
    elif filename.endswith(".tex"):
        # Use regular expressions to extract regular text from the LaTeX file
        filepath = os.path.join("Documents", filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
        # Find the line number of the \begin{document} command
        doc_start = None
        for i, line in enumerate(lines):
            if "\\begin{document}" in line:
                doc_start = i
                break
        # Ignore all lines before the \begin{document} command
        lines = lines[doc_start:]
        text = "".join(lines)
        
        # Add the text and title to the DataFrame
        title = os.path.splitext(filename)[0] # Remove the file extension from the filename
        df = df.append({"Title": title, "Text": text}, ignore_index=True)

# Loop through the rows and create overlapping chunks for each text
chunks = []
for i, row in df.iterrows():
    # Tokenize the text for the current row
    tokens = nltk.word_tokenize(row['Text'])

    # Loop through the tokens and create overlapping chunks
    for j in range(0, len(tokens), chunk_size - overlap_size):
        # Get the start and end indices of the current chunk
        start = j
        end = j + chunk_size

        # Create the current chunk by joining the tokens within the start and end indices
        chunk = ' '.join(tokens[start:end])
        
        # Add the article title to the beginning of the chunk
        chunk_with_title = "This text comes from the article titled " + row['Title'] + ". " + chunk
        
        # Append the current chunk to the list of chunks, along with the corresponding title
        chunks.append([row['Title'], chunk_with_title])

# Convert the list of chunks to a dataframe and write to a CSV
df_chunks = pd.DataFrame(chunks, columns=['Title', 'Text'])

article_embeddings = openai.Embedding.create(
      model=embeddingmodel,
      input=df_chunks.iloc[:, 1].tolist()
    )

df_chunks['article_embeddings'] = article_embeddings['data']

df_chunks.to_csv('textchunks.csv', encoding='utf-8', index=False)

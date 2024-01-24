#Import necessary libraries
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')
news = newsgroups.data

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(news)

# Cosine similarity function
def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity(vector1, vector2)

def find_similar_documents(input_doc, top_n=5):
    # Vectorizing the input document using the existing TF-IDF vectorizer
    input_vector = tfidf_vectorizer.transform([input_doc])
    
    # Calculating cosine similarity with all documents in the dataset
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix)
    
    # Get the top N similar documents
    similar_indices = cosine_similarities.argsort()[0][-top_n-1:-1][::-1]
    similar_documents = [(news[i], cosine_similarities[0, i]) for i in similar_indices]

    return similar_documents

# Example document (you can replace this with any text)
input_document = "With the help of Artificial Intelligence scientists have discovered a new antibiotic"

# Find similar documents
similar_docs = find_similar_documents(input_document, top_n=5)

# Display the results
for doc, score in similar_docs:
    print(f"Similarity Score: {score}\nDocument: {doc}\n")
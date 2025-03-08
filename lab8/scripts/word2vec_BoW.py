import os
import re
import sys
import spacy
import random
import warnings
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans

random.seed(42)
warnings.filterwarnings("ignore")


#* Function to preprocess the text
def preprocess_text(text):
    text = re.sub(r"^\[[A-Z]+\]\s*\n?", "", text) # remove the post type tag
    text = text.replace("\n", " ") # remove the new line characters
    text = re.sub(r"http\S+|www\.\S+", "", text) # remove the urls and non-alphabetic characters
    return text


#* Function to tokenize the text
def tokenize_text(text):
    #? Iterate through each sentence in the file
    data = []
    for sentence in sent_tokenize(text):
        tokens = []
        
        #? Tokenize the sentence into words
        for token in word_tokenize(sentence):
            token = token.lower()
            
            #? Remove stop words and non-alphabetic characters
            if token not in stop_words and token.isalpha():
                tokens.append(token)
        
        tokens = [token.lemma_ for token in nlp(" ".join(tokens))] # lemmatize the words
        data.append(tokens)
    
    return data


#* Function to load the data and preprocess it
def data_engineering(input_dir):
    files = os.listdir(input_dir)
    for file in files:
        file_id = file.split("_")[1].split(".")[0]
        
        with open(os.path.join(input_dir, file), "r") as f:
            text = f.read()
        
        processed_text = preprocess_text(text)
        tokens = tokenize_text(processed_text)
        
        data_dict[file_id] = {
            "tokens": tokens
        }
    
    return data_dict


#* Function to train the CBOW model
def train_CBOW_model(data_dict):
    #? Initialize the parameters
    kwargs = {
        "min_count": 1,
        "vector_size": 100,
        "window": 5,
    }
    
    corpus = []
    for key in data_dict:
        corpus.extend(data_dict[key]["tokens"])
    
    CBOW_model = Word2Vec(corpus, **kwargs)
    
    return CBOW_model


#* Function to get the word embeddings
def get_embeddings(model):
    words = model.wv.index_to_key
    embeddings = np.array([model.wv[word] for word in words])
    return words, embeddings


#* Function to perform clustering
def cluster_embeddings(embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.labels_


#* Function to vectorize the post based on the frequency of the words in each bin
def vectorize_post(tokens, word_cluster_mapping, n_clusters):
    all_words = [word for sentence in tokens for word in sentence]
    
    #? Initialize the histogram with the length of the number of clusters
    histogram = np.zeros(n_clusters)
    
    #? Loop through all the words and count the frequency of the words in each bin
    for word in all_words:
        if word in word_cluster_mapping:
            bin_idx = word_cluster_mapping[word]
            histogram[bin_idx] += 1
    
    #? Normalize the histogram
    total_words = np.sum(histogram)
    if total_words > 0:
        histogram = histogram / total_words
    
    return histogram


#* Main function
def main():
    #` Step 1: Load text files, preprocess the text, and tokenize the text
    data_dict = data_engineering(input_dir)
    
    #` Step 2: Train the CBOW model
    CBOW_model = train_CBOW_model(data_dict)
    
    #` Step 3: Get the word embeddings
    words, embeddings = get_embeddings(CBOW_model)
    
    #` Step 4: Perform clustering 
    cluster_labels = cluster_embeddings(embeddings, n_clusters=10)
    
    word_cluster_mapping = dict(zip(words, cluster_labels))
    
    #` Step 5: Vectorize the posts
    for post_id, content in data_dict.items():
        tokens = content["tokens"]
        doc_vector = vectorize_post(tokens, word_cluster_mapping, n_clusters=10)
        doc_vectors[post_id] = doc_vector
    
    return doc_vectors

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(BASE_DIR, "../data/raw/raw_data/")
    
    stop_words = set(stopwords.words("english"))
    nlp = spacy.load("en_core_web_sm")
    
    data_dict = {}
    doc_vectors = {}
    main()
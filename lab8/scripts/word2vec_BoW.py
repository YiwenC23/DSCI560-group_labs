import os
import re
import spacy
import random
import warnings
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans

seed = random.seed(42)
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
    data_dict = {}
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
def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
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
    
    return histogram, all_words


#* Main function
def main():
    #` Step 1: Load text files, preprocess the text, and tokenize the text
    data_dict = data_engineering(input_dir)
    
    #` Step 2: Train the CBOW model
    CBOW_model = train_CBOW_model(data_dict)
    
    #` Step 3: Get the word embeddings
    words, embeddings = get_embeddings(CBOW_model)
    
    #` Step 4: Perform clusterings with three different number of clusters
    cluster_labels_1 = cluster_embeddings(embeddings, n_clusters=50)
    cluster_labels_2 = cluster_embeddings(embeddings, n_clusters=100)
    cluster_labels_3 = cluster_embeddings(embeddings, n_clusters=200)
    
    word_cluster_mapping_1 = dict(zip(words, cluster_labels_1))
    word_cluster_mapping_2 = dict(zip(words, cluster_labels_2))
    word_cluster_mapping_3 = dict(zip(words, cluster_labels_3))
    
    #` Step 5: Vectorize the posts with three different number of clusters
    word2vec_BoW_tagged_doc = {}
    docVec_dict = {"dim1": {}, "dim2": {}, "dim3": {}}
    
    #? Get the vectorized posts for three different dimensions
    for post_id, content in data_dict.items():
        tokens = content["tokens"]
        doc_vectors_1, words_list = vectorize_post(tokens, word_cluster_mapping_1, n_clusters=50)
        doc_vectors_2, _ = vectorize_post(tokens, word_cluster_mapping_2, n_clusters=100)
        doc_vectors_3, _ = vectorize_post(tokens, word_cluster_mapping_3, n_clusters=200)
        
        docVec_dict["dim1"][post_id] = doc_vectors_1
        docVec_dict["dim2"][post_id] = doc_vectors_2
        docVec_dict["dim3"][post_id] = doc_vectors_3
    
        word2vec_BoW_tagged_doc[post_id] = " ".join(words_list)
    
    return docVec_dict, word2vec_BoW_tagged_doc


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(BASE_DIR, "../data/raw_data/")
    
    stop_words = set(stopwords.words("english"))
    nlp = spacy.load("en_core_web_sm")
    
    main()
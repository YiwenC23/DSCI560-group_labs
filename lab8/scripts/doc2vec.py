import os
import re
import spacy
import numpy as np
import random
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from sqlalchemy import select

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)

# Step 0: Pre-process
def pre_process():
    RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_data") # subject to change based on your folder organization
    stop_words = set(stopwords.words("english"))
    nlp = spacy.load("en_core_web_sm")
    posts = []
    doc2vec_tagged_docs = {}
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        if os.path.isfile(file_path):
            post_id = filename.split(".")[0].split("_")[1]  # Extract post ID from filename
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
                # Tokenize messages: "This is a sample document about Python programming!" --> ['this', 'is', 'sample', 'document', 'about', 'python', 'programming']
                tokens = simple_preprocess(text, deacc=True)
                tokens = [word for word in tokens if word not in stop_words]
                tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
                # Prepare tagged documents for doc2vec: ['this', 'is', 'sample', 'document', 'about', 'python', 'programming'] --> TaggedDocument(words=['this', 'is', 'sample', 'document', 'about', 'python', 'programming'], tags=['0'])
                posts.append(TaggedDocument(tokens, [post_id]))
                doc2vec_tagged_docs[post_id] = " ".join(tokens)
    return posts, doc2vec_tagged_docs
                
# Step 1: Train a Doc2Vec model for document embeddings
def train_doc2vec(posts, vector_size, min_count, epochs):
    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=seed_val)
    model.build_vocab(posts)
    model.train(posts, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Step 2: Generate embeddings for each document
def generate_embeddings(model, posts):
    embeddings = [model.infer_vector(doc.words) for doc in posts]
    return embeddings

def algorithm():
    # Preprocess data
    posts, doc2vec_tagged_docs = pre_process()
    
    # Train the model
    doc2vec_model_1 = train_doc2vec(posts, 5, 2, 20) # Works well for small datasets, captures basic word relationships.
    doc2vec_model_2 = train_doc2vec(posts, 10, 2, 20) # Balanced configuration for capturing meaningful embeddings.
    doc2vec_model_3 = train_doc2vec(posts, 20, 2, 20) # Avoids noise from rare words while maintaining good generalization.

    # Generate embeddings
    embeddings_1 = generate_embeddings(doc2vec_model_1, posts)
    embeddings_2 = generate_embeddings(doc2vec_model_2, posts)
    embeddings_3 = generate_embeddings(doc2vec_model_3, posts)

    # Organize the output
    embeddings_dict = {
        "config1": {post.tags[0]: embedding for post, embedding in zip(posts, embeddings_1)},
        "config2": {post.tags[0]: embedding for post, embedding in zip(posts, embeddings_2)},
        "config3": {post.tags[0]: embedding for post, embedding in zip(posts, embeddings_3)}
    }
    
    return embeddings_dict, doc2vec_tagged_docs

if __name__ == "__main__":
    algorithm()

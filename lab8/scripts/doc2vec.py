import os
import re
import spacy
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
# from database import SessionLocal, PostInfo
from sqlalchemy import select
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

# Step 1: Train a Doc2Vec model for document embeddings
def train_doc2vec(tagged_data, vector_size, min_count, epochs):
    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, seed=42)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Step 2: Generate embeddings for each document
def generate_embeddings(model, tagged_docs):
    embeddings = [model.infer_vector(doc.words) for doc in tagged_docs]
    return embeddings


def algorithm():
    # extract data
    # session = SessionLocal()
    # file_paths = session.execute(select(PostInfo.file_path)).scalars().all()
    RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_data")
    stop_words = set(stopwords.words("english"))
    nlp = spacy.load("en_core_web_sm")
    # rm_words_list = ["use", "data", "like", "just"]
    posts = []
    original_posts = {}
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        if os.path.isfile(file_path):
            post_id = filename.split(".")[0]  # Extract post ID from filename
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                original_posts[post_id] = text
                text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
                # Tokenize messages: "This is a sample document about Python programming!" --> ['this', 'is', 'sample', 'document', 'about', 'python', 'programming']
                tokens = simple_preprocess(text, deacc=True)
                tokens = [word for word in tokens if word not in stop_words]
                tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
                # tokens = [word for word in tokens if word not in rm_words_list]
                # Prepare tagged documents for doc2vec: ['this', 'is', 'sample', 'document', 'about', 'python', 'programming'] --> TaggedDocument(words=['this', 'is', 'sample', 'document', 'about', 'python', 'programming'], tags=['0'])
                posts.append(TaggedDocument(tokens, [post_id]))

    # Train the model
    doc2vec_model_1 = train_doc2vec(posts, 50, 1, 100) # Works well for small datasets, captures basic word relationships.
    doc2vec_model_2 = train_doc2vec(posts, 100, 2, 50) # Balanced configuration for capturing meaningful embeddings.
    doc2vec_model_3 = train_doc2vec(posts, 200, 5, 30) # Avoids noise from rare words while maintaining good generalization.

    # Generate embeddings
    embeddings_1 = generate_embeddings(doc2vec_model_1, posts)
    embeddings_2 = generate_embeddings(doc2vec_model_2, posts)
    embeddings_3 = generate_embeddings(doc2vec_model_3, posts)



if __name__ == "__main__":
    algorithm()


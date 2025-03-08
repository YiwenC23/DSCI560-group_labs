import os
import glob
import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

# Step 1: Read all .txt files
def load_text_files(folder_path):
    folder_path = "/home/fariha/Downloads/lab8_raw_data/raw_data"
    text_data = []
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data.append(file.read().strip())
    
    return text_data

# Step 2: Preprocess text (Tokenization)
def preprocess_text(text_data):
    processed_data = [gensim.utils.simple_preprocess(doc) for doc in text_data]
    return processed_data

# Step 3: Generate Doc2Vec Embeddings with different configurations
def train_doc2vec_models(tagged_data):
    models = {}

    config_settings = {
        "config1": {"vector_size": 50, "min_count": 2, "epochs": 20},
        "config2": {"vector_size": 100, "min_count": 5, "epochs": 30},
        "config3": {"vector_size": 200, "min_count": 3, "epochs": 40}
    }

    for config_name, params in config_settings.items():
        model = Doc2Vec(vector_size=params["vector_size"], min_count=params["min_count"], epochs=params["epochs"])
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        models[config_name] = model

    return models

# Step 4: Cluster the embeddings using K-Means
def cluster_embeddings(model, num_clusters=5):
    vector_size = model.vector_size  # Ensure correct vector size
    vectors = [model.dv[i] for i in range(len(model.dv))]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vectors)
    return clusters, vectors, vector_size  # Return vector_size for consistency

# Step 5: Find the closest cluster for a given input
def find_closest_cluster(user_input, model, vectors, clusters, vector_size):
    input_vector = model.infer_vector(gensim.utils.simple_preprocess(user_input))

    # Ensure input_vector has the correct shape
    if len(input_vector) != vector_size:
        print(f"Error: Mismatched vector sizes. Expected {vector_size}, got {len(input_vector)}")
        return

    min_distance = float("inf")
    closest_cluster = None

    for i, vector in enumerate(vectors):
        distance = cosine(input_vector, vector)
        if distance < min_distance:
            min_distance = distance
            closest_cluster = clusters[i]

    print(f"\nClosest Cluster: {closest_cluster}")

# Main Execution
def algorithm():
    data_folder = "/home/fariha/Downloads/lab8_raw_data/raw_data"  # Change this to your dataset path
    raw_texts = load_text_files(data_folder)
    processed_texts = preprocess_text(raw_texts)

    tagged_data = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(processed_texts)]
    models = train_doc2vec_models(tagged_data)

    for config_name, model in models.items():
        print(f"\nClustering results for {config_name}:")
        clusters, embeddings = cluster_embeddings(model)
        print(clusters)

    # User input for finding the closest cluster
    while True:
        user_input = input("\nEnter a keyword or message to find the closest cluster (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        find_closest_cluster(user_input, models["config1"], embeddings, clusters)  # Using config1 model for search

if __name__ == "__main__":
    algorithm()


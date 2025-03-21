import os
import re
import spacy
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from database import SessionLocal, PostInfo
from sqlalchemy import select

np.random.seed(42)

# Step 1: Train a Doc2Vec model for document embeddings
def train_doc2vec(tagged_docs):
    # Tokenize messages: "This is a sample document about Python programming!" --> ['this', 'is', 'sample', 'document', 'about', 'python', 'programming']
    # tokenized_messages = [simple_preprocess(message) for message in tagged_docs]

    # Prepare tagged documents for doc2vec: ['this', 'is', 'sample', 'document', 'about', 'python', 'programming'] --> TaggedDocument(words=['this', 'is', 'sample', 'document', 'about', 'python', 'programming'], tags=['0'])
    # tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokenized_messages)]

    # Train doc2vec model
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40, seed=42)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

    return model

# Step 2: Generate embeddings for each document
def generate_embeddings(model, tagged_docs):
    embeddings = [model.infer_vector(doc.words) for doc in tagged_docs]
    return embeddings

# Step 3: Cluster the Embeddings
def perform_clustering(embeddings, max_clusters):
    best_score = -1  # Silhouette Score ranges from -1 to 1
    optimal_clusters = 2  # At least 2 clusters are needed for Silhouette Score
    best_kmeans = None
    best_clusters = None

    # Evaluate Silhouette Score for different numbers of clusters
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, clusters)

        # Update the best score and corresponding model
        if score > best_score:
            best_score = score
            optimal_clusters = i
            best_kmeans = kmeans
            best_clusters = clusters

    return best_clusters, best_kmeans, optimal_clusters

# Step 4: Extract Keywords for Each Cluster
def extract_keywords(tagged_docs, clusters):
    cluster_messages = {i: [] for i in range(max(clusters) + 1)}
    for i, cluster in enumerate(clusters):
        cluster_messages[cluster].append(tagged_docs[i])

    # Extract keywords using TF-IDF
    cluster_keywords = {}
    for cluster, messages_in_cluster in cluster_messages.items():
        messages_str = [" ".join(doc.words) if hasattr(doc, "words") else doc for doc in messages_in_cluster]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(messages_str)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-5:][::-1]]
        # print([feature_names[i] for i in tfidf_scores.argsort()[::-1]])
        cluster_keywords[cluster] = top_keywords

    return cluster_keywords

# Step 5: Visualize the clusters
def visualize_clusters(embeddings, clusters, cluster_keywords, messages, tagged_data, top_n_samples=3, highlight_cluster = None):
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(10, 8))

    # Highlight the closest cluster
    if highlight_cluster is not None:
        highlight_indices = [i for i, c in enumerate(clusters) if c == highlight_cluster]
        plt.scatter(reduced_embeddings[highlight_indices, 0], reduced_embeddings[highlight_indices, 1], c='grey', s=100, edgecolor='black', linewidth=2, label=f'Cluster {highlight_cluster} (Closest)', alpha=0.8)
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', alpha=0.5, label='Other Clusters')

    # Plot all clusters
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', alpha=0.8)

    # Annotate clusters with top keywords
    for cluster in cluster_keywords:
        # Get the centroid of the cluster
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
        centroid = reduced_embeddings[cluster_indices].mean(axis=0)

        # Annotate the centroid with top keywords
        keywords = ", ".join(cluster_keywords[cluster])

        # Place the annotation
        plt.text(centroid[0], centroid[1], f"Cluster {cluster}\nKeywords: {keywords}", fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

    # Add legend and labels
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Cluster Visualization with Keywords")
    plt.show()

    # Verify cluster similarity by displaying sample messages
    if highlight_cluster is None:
        print("\nVerifying Cluster Similarity:")
        cluster_messages = {i: [] for i in range(max(clusters) + 1)}
        for i, cluster in enumerate(clusters):
            post_id = tagged_data[i].tags[0]
            cluster_messages[cluster].append(messages[post_id])

        for cluster, messages_in_cluster in cluster_messages.items():
            print(f"\nCluster {cluster} Sample Messages:")
            for message in messages_in_cluster[:top_n_samples]:
                print(f" - {message}")

# Find the closest cluster based on the user input
def find_closest_cluster(input_message, doc2vec_model, embeddings, clusters, cluster_keywords, tagged_data, messages):
    # Preprocess the input message
    cleaned_input = simple_preprocess(input_message)

    # Generate embedding for the input message
    input_embedding = doc2vec_model.infer_vector(cleaned_input)

    # Find the closest cluster
    closest_cluster = None
    min_distance = float('inf')
    for i, embedding in enumerate(embeddings):
        distance = np.linalg.norm(np.array(embedding) - np.array(input_embedding))
        if distance < min_distance:
            min_distance = distance
            closest_cluster = clusters[i]
                
    cluster_messages = {i: [] for i in range(max(clusters) + 1)}
    for i, cluster in enumerate(clusters):
        if cluster == closest_cluster:
            post_id = tagged_data[i].tags[0]
            cluster_messages[cluster].append(messages[post_id])
    for cluster, messages_in_cluster in cluster_messages.items():
        print(f"\nCluster {cluster} Sample Messages:")
        for message in messages_in_cluster[:3]:
            print(f" - {message}")

    # Visualize the clusters with the closest cluster highlighted
    visualize_clusters(embeddings, clusters, cluster_keywords, messages, tagged_data, highlight_cluster=closest_cluster)

def algorithm():
    # extract data
    session = SessionLocal()
    file_paths = session.execute(select(PostInfo.file_path)).scalars().all()
    stop_words = set(stopwords.words("english"))
    nlp = spacy.load("en_core_web_sm")
    rm_words_list = ["use", "data", "like", "just"]
    posts = []
    original_posts = {}
    for file_path in file_paths:
        post_id = file_path.split("/")[-1].split(".")[0]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            original_posts[post_id] = text
            text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
            tokens = simple_preprocess(text, deacc=True)
            tokens = [word for word in tokens if word not in stop_words]
            tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
            tokens = [word for word in tokens if word not in rm_words_list]
            posts.append(TaggedDocument(tokens, [post_id]))
            tagged_data = posts

    # Train the model
    doc2vec_model = train_doc2vec(posts)

    # Generate embeddings
    embeddings = generate_embeddings(doc2vec_model, posts)

    # Use KMeans to do clustering
    clusters, kmeans, optimal_clusters = perform_clustering(embeddings, max_clusters=min(10, len(embeddings)-1))

    # Extract keywords for each cluster
    cluster_keywords = extract_keywords(posts, clusters)

    # Visualize the cluster
    visualize_clusters(embeddings, clusters, cluster_keywords, original_posts, tagged_data, top_n_samples=3)

    while True:
        user_input = input("\nEnter a keyword or message to find the closest cluster (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        find_closest_cluster(user_input, doc2vec_model, embeddings, clusters, cluster_keywords, tagged_data, original_posts)



if __name__ == "__main__":
    # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # stop_words_path = os.path.join(CURRENT_DIR, "./test/output/stopwords/")
    # stop_words = set(stopwords.words("english"))
    algorithm()


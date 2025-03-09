import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer

from doc2vec import algorithm
from word2vec_BoW import main

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)


#* Step 1: Cluster the Embeddings Using Cosine Distance
def perform_clustering(embeddings, max_clusters=10):
    best_score = -1  # Silhouette Score ranges from -1 to 1
    optimal_clusters = 2  # At least 2 clusters needed
    best_kmeans = None
    best_clusters = None
    
    # When we normalize the embeddings to unit vectors, the Euclidean distance in this space becomes equivalent to cosine distance
    embeddings_norm = normalize(embeddings, norm="l2")
    
    # Find the best number of clusters
    for k in range(2, max_clusters + 1):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed_val, batch_size=256, n_init=50)
        clusters = kmeans.fit_predict(embeddings_norm)
        
        # Compute silhouette score using cosine distance
        score = silhouette_score(embeddings_norm, clusters, metric="cosine", random_state=seed_val)
        
        if score > best_score:
            best_score = score
            optimal_clusters_number = k
            best_kmeans = kmeans
            best_clusters = clusters
    
    return best_clusters, best_kmeans, optimal_clusters_number, best_score


#* Step 2: Extract keywords from the centroids and TF-IDF algorithm
def extract_keywords_from_centroids(embeddings, best_clusters, tagged_docs, best_kmeans):
    if isinstance(tagged_docs, dict):
        tagged_docs = [tagged_docs[k] for k in sorted(tagged_docs.keys())]
    cluster_keywords = {}
    
    for cluster_id in range(best_kmeans.n_clusters):
        centroid = best_kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        
        cluster_indices = np.where(best_clusters == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        distances = cosine_distances(cluster_embeddings, centroid).flatten()
        
        closest_doc_idx = cluster_indices[np.argmin(distances)]
        closest_doc_text = tagged_docs[closest_doc_idx]
        
        # Use TF-IDF to extract keywords
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([closest_doc_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().flatten()
        
        top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-5:][::-1]]
        cluster_keywords[cluster_id] = top_keywords
    
    return cluster_keywords


#* Step 3: Visualize the clusters
def visualize_clusters(embeddings, best_clusters, cluster_keywords):
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2, random_state=seed_val)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=best_clusters, cmap="viridis", alpha=0.8)

    # Set annotation position in bottom-left corner
    x_min, y_min = reduced_embeddings.min(axis=0)
    annotation_x = x_min - 1  # Shift annotations a bit to the left
    annotation_y_start = y_min - 1  # Start annotation at the bottom
    
    # Calculate dynamic spacing based on the number of clusters to avoid overlap
    total_clusters = len(cluster_keywords)
    line_spacing = 1.2  # Increased line spacing to prevent vertical overlap
    column_spacing = 3.0  # Increased column spacing for better separation
    
    max_lines = 10  # Maximum lines before shifting to a new column
    num_columns = (total_clusters // max_lines) + 1  # Calculate the number of columns needed
    
    # # Annotate clusters sequentially in the bottom-left corner
    # for i, cluster in enumerate(sorted(cluster_keywords.keys())):
    #     # Calculate y position
    #     column_index = i // max_lines
    #     row_index = i % max_lines
    #     annotation_y = annotation_y_start + row_index * line_spacing
        
    #     # Adjust x position based on column index to avoid overlap in multiple columns
    #     annotation_x_shift = annotation_x + column_index * column_spacing  # Shift to the right for next column
        
    #     keywords = ", ".join(cluster_keywords[cluster])
    #     plt.text(annotation_x_shift, annotation_y,  
    #              f"Cluster {cluster}: {keywords}",
    #              fontsize=10, ha="left", va="bottom", bbox=dict(facecolor="white", alpha=0.8))
    
    # Add legend and labels
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Cluster Visualization with Keywords")
    plt.show()


#* Main function to run the complete analysis
def evaluate_embeddings(best_clusters, optimal_clusters_number, best_score, embeddings, cluster_keywords, best_kmeans):
    # Use Silhouette Score to measure how well-separated the clusters are.
    # Higher Silhouette Score → Better clustering (embeddings are well-structured)
    print(f"Optimal number of clusters: {optimal_clusters_number}")
    print(f"Best Silhouette Score: {best_score}")
    
    # Check Cluster Distribution
    # Good embeddings → Evenly distributed clusters
    unique, counts = np.unique(best_clusters, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    print(f"Cluster Distribution: {cluster_distribution}")
    
    # Check Cluster Centroids
    # Well-separated centroids → embeddings capture distinct features well.
    pca = PCA(n_components=2, random_state=seed_val)
    centroids_2d = pca.fit_transform(best_kmeans.cluster_centers_)
    # Plot the centroids
    plt.figure(figsize=(8, 6))
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="red", marker="x", s=200)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Cluster Centroids in 2D (PCA Projection)")
    plt.grid()
    plt.show()
    
    # Check Representative Samples
    # If embeddings are good, samples from the same cluster should be semantically similar.
    print("Representative samples from each cluster:")
    for cluster_id, samples in cluster_keywords.items():
        print(f"Cluster {cluster_id}: {samples}")
    
    # Visualize clusters
    # Clear separations → good embeddings.
    visualize_clusters(embeddings, best_clusters, cluster_keywords)


def workflow():
    #` Clustering for Word2Vec-BoW with three different dimensions
    word2vec_BoW_dict, word2vec_BoW_tagged_doc = main()
    
    for dim, emb_dict in word2vec_BoW_dict.items():
        if dim == "dim1":
            dim = 5
        elif dim == "dim2":
            dim = 10
        elif dim == "dim3":
            dim = 20
        print(f"\nProcessing Word2Vec+BoW embeddings for dimension: {dim}")
        
        embeddings = np.array(list(emb_dict.values()))
        best_clusters, best_kmeans, optimal_clusters_number, best_score = perform_clustering(embeddings, max_clusters=10)
        cluster_keywords = extract_keywords_from_centroids(embeddings, best_clusters, word2vec_BoW_tagged_doc, best_kmeans)
        evaluate_embeddings(best_clusters, optimal_clusters_number, best_score, embeddings, cluster_keywords, best_kmeans)
    
    #` Clustering for Doc2Vec with three different dimensions
    doc2vec_dict, doc2vec_tagged_doc = algorithm()
    
    for dim, emb_dict in doc2vec_dict.items():
        if dim == "config1":
            dim = 5
        elif dim == "config2":
            dim = 10
        elif dim == "config3":
            dim = 20
        print(f"\nProcessing Doc2Vec embeddings for dimension: {dim}")
        
        embeddings = np.array(list(emb_dict.values()))
        best_clusters, best_kmeans, optimal_clusters_number, best_score = perform_clustering(embeddings, max_clusters=10)
        cluster_keywords =  extract_keywords_from_centroids(embeddings, best_clusters, doc2vec_tagged_doc, best_kmeans)
        evaluate_embeddings(best_clusters, optimal_clusters_number, best_score, embeddings, cluster_keywords, best_kmeans)


if __name__ == "__main__":
    workflow()

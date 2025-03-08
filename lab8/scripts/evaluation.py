

# Step 3: Cluster the Embeddings Using Cosine Distance
def perform_clustering(embeddings, max_clusters=10):
    best_score = -1  # Silhouette Score ranges from -1 to 1
    optimal_clusters = 2  # At least 2 clusters needed
    best_kmeans = None
    best_clusters = None

    # When we normalize the embeddings to unit vectors, the Euclidean distance in this space becomes equivalent to cosine distance
    embeddings_norm = normalize(embeddings, norm='l2')

    # Find the best number of clusters
    for k in range(2, max_clusters + 1):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=10)
        clusters = kmeans.fit_predict(embeddings_norm)
        
        # Compute silhouette score using cosine distance
        score = silhouette_score(embeddings_norm, clusters, metric="cosine")
        
        if score > best_score:
            best_score = score
            optimal_clusters_number = k
            best_kmeans = kmeans
            best_clusters = clusters

    return best_clusters, best_kmeans, optimal_clusters_number, best_score


# Step 4: Extract Keywords for Each Cluster
def extract_keywords(tagged_docs, best_clusters):
    cluster_messages = {i: [] for i in range(max(best_clusters) + 1)}
    for i, cluster in enumerate(best_clusters):
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
def visualize_clusters(embeddings, best_clusters, cluster_keywords):
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=best_clusters, cmap='viridis', alpha=0.8)

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

    # Annotate clusters sequentially in the bottom-left corner
    for i, cluster in enumerate(sorted(cluster_keywords.keys())):
        # Calculate y position
        column_index = i // max_lines
        row_index = i % max_lines
        annotation_y = annotation_y_start + row_index * line_spacing
        
        # Adjust x position based on column index to avoid overlap in multiple columns
        annotation_x_shift = annotation_x + column_index * column_spacing  # Shift to the right for next column
        
        keywords = ", ".join(cluster_keywords[cluster])
        plt.text(annotation_x_shift, annotation_y,  
                 f"Cluster {cluster}: {keywords}",
                 fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

    
    # Add legend and labels
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Cluster Visualization with Keywords")
    plt.show()


# Main function to run the complete analysis
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
    pca = PCA(n_components=2)
    centroids_2d = pca.fit_transform(best_kmeans.cluster_centers_)
    # Plot the centroids
    plt.figure(figsize=(8, 6))
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200)
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
    
if __name__ == "__main__":
    # Use KMeans to do clustering
    best_clusters_1, best_kmeans_1, optimal_clusters_number_1, best_score_1 = perform_clustering(embeddings_1, max_clusters=min(10, len(embeddings_1)-1))
    best_clusters_2, best_kmeans_2, optimal_clusters_number_2, best_score_2 = perform_clustering(embeddings_2, max_clusters=min(10, len(embeddings_2)-1))
    best_clusters_3, best_kmeans_3, optimal_clusters_number_3, best_score_3 = perform_clustering(embeddings_3, max_clusters=min(10, len(embeddings_3)-1))

    # Extract keywords for each cluster
    cluster_keywords_1 = extract_keywords(posts, best_clusters_1)
    cluster_keywords_2 = extract_keywords(posts, best_clusters_2)
    cluster_keywords_3 = extract_keywords(posts, best_clusters_3)
    
    # Evaluate the clustering
    print("Configuration 1")
    evaluate_embeddings(best_clusters_1, optimal_clusters_number_1, best_score_1, embeddings_1, cluster_keywords_1, best_kmeans_1)
    print("")
    print("Configuration 2")
    evaluate_embeddings(best_clusters_2, optimal_clusters_number_2, best_score_2, embeddings_2, cluster_keywords_2, best_kmeans_2)
    print("")
    print("Configuration 3")
    evaluate_embeddings(best_clusters_3, optimal_clusters_number_3, best_score_3, embeddings_3, cluster_keywords_3, best_kmeans_3)

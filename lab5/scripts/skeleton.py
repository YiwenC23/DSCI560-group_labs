from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mysql.connector

df = pd.read_parquet("/home/fariha/fariha_9625734353/DSCI560-group_labs/lab5/data/processed_data/reddit_datascience.parquet")
messages = df["content"].tolist()

# Part 4a: Message Content Abstraction
class MessageEmbedding:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.model = None
    def train_doc2vec(self, messages):
        # Prepare documents for training
        tagged_data = [
            TaggedDocument(words=word_tokenize(msg.lower()), 
                         tags=[str(i)]) 
            for i, msg in enumerate(messages)
        ]
        
        # Train Doc2Vec model
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            min_count=2,
            epochs=40
        )
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, 
                        total_examples=self.model.corpus_count, 
                        epochs=self.model.epochs)
    
    def get_vector(self, message):
        # Convert single message to vector
        return self.model.infer_vector(word_tokenize(message.lower()))
    
    def save_vectors_to_db(self, messages, db_connection):
        # Save vectors to database
        cursor = db_connection.cursor()
        for msg_id, message in enumerate(messages):
            vector = self.get_vector(message)
            # Assuming you have a table 'message_vectors'
            query = """
            UPDATE messages 
            SET vector = %s 
            WHERE id = %s
            """
            cursor.execute(query, (vector.tostring(), msg_id))
        db_connection.commit()

# Part 4b: Message Clustering
class MessageClustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.cluster_labels = None
        self.cluster_centers = None
    
    def cluster_messages(self, vectors):
        # Perform K-means clustering
        self.cluster_labels = self.kmeans.fit_predict(vectors)
        self.cluster_centers = self.kmeans.cluster_centers_
        return self.cluster_labels
    
    def get_cluster_keywords(self, messages, cluster_id):
        # Get messages from specific cluster and extract common keywords
        cluster_messages = [
            msg for i, msg in enumerate(messages) 
            if self.cluster_labels[i] == cluster_id
        ]
        # Add your keyword extraction logic here
        return cluster_messages
    
    def visualize_clusters(self, vectors):
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        for i in range(self.n_clusters):
            cluster_points = vectors_2d[self.cluster_labels == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f'Cluster {i}'
            )
        plt.title('Message Clusters Visualization')
        plt.legend()
        plt.show()

def main():
    # Connect to database
    db_connection = mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="your_database"
    )
    # Fetch messages from database
    cursor = db_connection.cursor()
    cursor.execute("SELECT message FROM messages")
    messages = [row[0] for row in cursor.fetchall()]
    # Create embeddings
    embedder = MessageEmbedding(vector_size=100)
    embedder.train_doc2vec(messages)
    # Get vectors for all messages
    vectors = np.array([
        embedder.get_vector(message) 
        for message in messages
    ])    
    # Save vectors to database
    embedder.save_vectors_to_db(messages, db_connection)
    # Perform clustering
    clusterer = MessageClustering(n_clusters=5)
    cluster_labels = clusterer.cluster_messages(vectors)
    # Visualize results
    clusterer.visualize_clusters(vectors)
    # Get keywords for each cluster
    for i in range(clusterer.n_clusters):
        cluster_messages = clusterer.get_cluster_keywords(messages, i)
        print(f"\nCluster {i} sample messages:")
        print(cluster_messages[:3])  # Show first 3 messages from cluster
if __name__ == "__main__":
    main()

import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# load spacy model
nlp = spacy.load("en_core_web_sm")

class MessageEmbedding:
    def __init__(self, vector_size=50):
        self.vector_size = vector_size
        self.model = None
    
    def train_doc2vec(self, messages):
        tagged_data = [TaggedDocument(words=[token.text for token in nlp(msg.lower())], tags=[str(i)]) for i, msg in enumerate(messages)]
        self.model = Doc2Vec(vector_size=self.vector_size, min_count=2, epochs=40)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
    
    def get_vector(self, message):
        return self.model.infer_vector([token.text for token in nlp(message.lower())])

# load your processed data
data_path = "/home/fariha/fariha_9625734353/DSCI560-group_labs/lab5/data/processed_data/reddit_datascience.parquet"
df = pd.read_parquet(data_path)

# use the 'content' column instead of 'message'
messages = df['content'].dropna().tolist()  # Ensure no missing values

# train embedding model
message_embedding = MessageEmbedding()
message_embedding.train_doc2vec(messages)

# convert messages to vectors
message_vectors = {i: message_embedding.get_vector(msg) for i, msg in enumerate(messages)}
vectors = np.array(list(message_vectors.values()))

# perform clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(vectors)
labels = kmeans.labels_

# reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=labels, palette='viridis', s=100)

# adding labels to the clusters
for i in range(num_clusters):
    plt.text(np.mean(reduced_vectors[labels == i, 0]), np.mean(reduced_vectors[labels == i, 1]),
             f'Cluster {i}', fontsize=12, ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.title("Clusters of Messages")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("clusters.png")  # Save the figure as clusters.png
#plt.show()

# identifying keywords for each cluster using tf-idf
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(messages)
feature_names = vectorizer.get_feature_names_out()

# extracting top keywords for each cluster
top_n_keywords = 5
for i in range(num_clusters):
    print(f"\nCluster {i}:")
    cluster_indices = np.where(labels == i)[0]
    cluster_tfidf_matrix = tfidf_matrix[cluster_indices]
    mean_tfidf = np.mean(cluster_tfidf_matrix, axis=0).flatten()
    top_indices = mean_tfidf.argsort()[-top_n_keywords:][::-1]
    
    print("Top keywords:", [feature_names[idx] for idx in top_indices])
    
    print("Messages in this cluster:")
    for idx in cluster_indices:
        print(f" - {messages[idx]}")

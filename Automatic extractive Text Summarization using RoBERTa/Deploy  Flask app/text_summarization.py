import nltk
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix 
from sentence_transformers import SentenceTransformer
from nltk.cluster import KMeansClusterer
import spacy

model = SentenceTransformer('stsb-roberta-base')

def get_sent_embeddings(sent):
	embeddings = model.encode([sent])
	return embeddings[0]

def distance_from_centroid(row):
  dist_matrix = distance_matrix([row['embeddings']], [row['Centroid'].tolist()])[0][0]
  return dist_matrix

def text_summarizer(raw_text):
	# Download punkt
	nltk.download('punkt')
	# Tokenize the raw_text into sentences
	sentences = nltk.sent_tokenize(raw_text)
	# Strip leading & trailing edge spaces
	sentences = [sentence.strip() for sentence in sentences]

	df = pd.DataFrame(sentences, columns = ['sentences'])
	#Initialize the model
	#model = SentenceTransformer('stsb-roberta-base')
	# Get sentence embeddings from model
	df['embeddings'] = df['sentences'].apply(get_sent_embeddings)

	# K-Means clustering
	n_clusters = int(len(df)/3)
	iterations = 25
	X = np.array(df['embeddings'].tolist())

	kcluster = KMeansClusterer(n_clusters, distance = nltk.cluster.util.cosine_distance,       # Cosine distance measure the distance/similarity between 2 vectors
                           repeats = iterations, avoid_empty_clusters = True)

	assigned_clusters = kcluster.cluster(X, assign_clusters = True)

	df['Cluster'] = assigned_clusters 
	df['Centroid'] = df['Cluster'].apply(lambda x : kcluster.means()[x])
	# Calculate the distance of each sentence embeddings from centroid
	df['distance_from_centroid'] = df.apply(distance_from_centroid, axis=1)
	sents = df.sort_values(by = 'distance_from_centroid', ascending=False).groupby('Cluster')
	# Get a sentence from each cluster which is closest to centroid
	sents = sents.head(1)['sentences'].tolist()
	# Join all sentences to get final summary
	summary = ' '.join(sents)  
	return summary
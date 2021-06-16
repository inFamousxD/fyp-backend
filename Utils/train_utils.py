from sklearn.cluster import KMeans
import torch

def get_class_wise_indices(dataset, label):
	# 1 -> Malware
	# 0 -> Benign
	indices = []
	for i in range(len(dataset)):
		if dataset[i]['label'] == label:
			indices.append(i)
	return indices
def get_subset_labelwise(dataset, label):
	indices = get_class_wise_indices(dataset, label)
	return torch.utils.data.Subset(dataset, indices)

def cluster_similar_points(kmeans, feature_vectors, dataset):
	#returns closest benign centroids as the "real vector" in GAN
	return torch.tensor([list(kmeans.cluster_centers_[i]) for i in kmeans.predict(feature_vectors)])
import os
import cv2
import numpy as np
import pickle
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def load_models(output_dir):
    """Load saved models and feature vectors."""
    features = np.load(os.path.join(output_dir, 'extracted_features.npy'))
    labels = np.load(os.path.join(output_dir, 'image_labels.npy'))
    
    with open(os.path.join(output_dir, 'image_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(output_dir, 'image_pca.pkl'), 'rb') as f:
        pca = pickle.load(f)

    return scaler, pca, features, labels

def compute_distances(features, labels):
    """Compute intra-class and inter-class distances."""
    intra_class_distances = []
    inter_class_distances = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            dist = distance.euclidean(features[i], features[j])
            if labels[i] == labels[j]:
                intra_class_distances.append(dist)
            else:
                inter_class_distances.append(dist)
    
    return intra_class_distances, inter_class_distances

def plot_histograms(intra_class_distances, inter_class_distances):
    """Plot histograms of intra-class and inter-class distances."""
    plt.figure(figsize=(12, 6))

    # Convert lists to numpy arrays if they aren't already
    intra_class_distances = np.array(intra_class_distances)
    inter_class_distances = np.array(inter_class_distances)

    # Intra-class distances plot
    plt.subplot(1, 2, 1)
    if len(intra_class_distances) > 0:
        plt.hist(intra_class_distances, bins=30, color='blue', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Intra-Class Distances')

    # Inter-class distances plot
    plt.subplot(1, 2, 2)
    if len(inter_class_distances) > 0:
        plt.hist(inter_class_distances, bins=30, color='red', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Inter-Class Distances')

    plt.tight_layout()
    plt.show()

def main():
    output_dir = r'E:\face matching using cbir\output'
    scaler, pca, features, labels = load_models(output_dir)

    intra_class_distances, inter_class_distances = compute_distances(features, labels)
    
    plot_histograms(intra_class_distances, inter_class_distances)
    
if __name__ == "__main__":
    main()

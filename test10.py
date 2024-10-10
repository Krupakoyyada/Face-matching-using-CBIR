import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt

# Suitable for desktop 
matplotlib.use('TkAgg')

# Function to preprocess the image using resizing
def pre_process(path, size=(224, 224)):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image at {path} could not be loaded.")
    image = cv2.resize(image, size)
    image = image.astype('float32') / 255.0
    return image

# Function to extract color histogram
def color(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to extract texture features using gabor
def gabor(image):
    from skimage.filters import gabor
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_features = []
    for theta in range(4):  # angles: 0, 45, 90, 135
        theta = theta / 4. * np.pi
        real, _ = gabor(gray_image, frequency=0.6, theta=theta)
        gabor_features.append(real.flatten())
    return np.concatenate(gabor_features)

# Function to extract shapes using canny edges
def shape(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.convertScaleAbs(gray_image)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges.flatten()

# Function to extract brightness feature
def brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_image)

def extract_features(image_path):
    image = pre_process(image_path)
    
    # Extract features using the same methods as in training
    color_h = color(image)
    gabor_f = gabor(image)
    edge_f = shape(image)
    brightness_f = np.array([brightness(image)])
    
    features = np.concatenate([color_h, gabor_f, edge_f, brightness_f])
    
    print(f"features size: {features.shape}")  # Print the size 
    return features

def load_models(output_dir):
    """Load models"""
    features = np.load(os.path.join(output_dir, 'extracted_features.npy'))
    labels = np.load(os.path.join(output_dir, 'image_labels.npy'))
    
    with open(os.path.join(output_dir, 'image_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(output_dir, 'image_pca.pkl'), 'rb') as f:
        pca = pickle.load(f)

    return scaler, pca, features, labels

def process_query_image(query_image_path, scaler, pca, feature_vectors, labels, distance_threshold=4):
    """Process the query image, find matches below  threshold."""
    # Extract features from the query image
    features = extract_features(query_image_path)
    
    # Apply PCA first and then scale the features using the scaler model
    reduced_features = pca.transform(features.reshape(1, -1))
    normalized_features = scaler.transform(reduced_features)

    # Compute distances
    distances = [distance.euclidean(normalized_features.flatten(), fv.flatten()) for fv in feature_vectors]
    
    # Filter matches based on the distance threshold
    filtered_indices = [i for i, dist in enumerate(distances) if dist < distance_threshold]
    filtered_distances = np.array(distances)[filtered_indices]
    filtered_labels = np.array(labels)[filtered_indices]
    
    return filtered_indices, filtered_distances, filtered_labels

def display_results(query_image_path, filtered_indices, filtered_distances, filtered_labels, database_dir):
    """Display the query image along with similar matches below threshold"""
    try:
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            raise ValueError(f"Query image not found: {query_image_path}")

        print(f"Displaying results: {query_image_path}")

        max_matches_to_display = 15
        num_matches = len(filtered_indices)
        cols = 4
        rows = (min(max_matches_to_display, num_matches) // cols) + 1  # Calculate rows needed for matches

        plt.figure(figsize=(12, 4 * rows))
        
        # Display the query image
        plt.subplot(rows, cols, 1)
        plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        plt.title("Query Image")
        plt.axis('off')

        if num_matches == 0:
            plt.text(0.5, 0.5, 'No similar images found', fontsize=12, ha='center', va='center')
        else:
            # Display filtered matches
            for i, index in enumerate(filtered_indices[:max_matches_to_display]):
                label = filtered_labels[i]
                image_path = os.path.join(database_dir, label, os.listdir(os.path.join(database_dir, label))[0])
                print(f"Loading: {image_path}")
                match_image = cv2.imread(image_path)
                if match_image is not None:
                    ax = plt.subplot(rows, cols, i + 2)
                    ax.imshow(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"Match {i+1}")
                    ax.axis('off')
                    plt.text(0.5, -0.2, f"Distance: {filtered_distances[i]:.2f}", fontsize=10, ha='center', va='center', transform=ax.transAxes)
                else:
                    print(f"Failed to load: {image_path}")

        plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust spacing between images
        plt.show()
    except Exception as e:
        print(f"Error displaying: {e}")

def main():
    output_dir = r'E:\face matching using cbir\output'
    query_dir = r'E:\face matching using cbir\testing'
    database_dir = r'E:\face matching using cbir\training'

    scaler, pca, feature_vectors, labels = load_models(output_dir)

    distance_threshold = 10  # Set your distance threshold here

    for person_name in os.listdir(query_dir):
        person_dir = os.path.join(query_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for query_image_name in os.listdir(person_dir):
            query_image_path = os.path.join(person_dir, query_image_name)
            if not (query_image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))):
                continue  # Skipping non-image files

            filtered_indices, filtered_distances, filtered_labels = process_query_image(query_image_path, scaler, pca, feature_vectors, labels, distance_threshold)
            
            print(f"Query Image: {query_image_name}")
            print("Matches:")
            
            for i, (index, distance, label) in enumerate(zip(filtered_indices, filtered_distances, filtered_labels)):
                print(f" {i+1}:")
                print(f" {label}")
                print(f"Distance: {distance:.2f}")
                print()
            
            display_results(query_image_path, filtered_indices, filtered_distances, filtered_labels, database_dir)

if __name__ == "__main__":
    main()



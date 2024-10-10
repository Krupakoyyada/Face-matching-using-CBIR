import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

def extract_features(image_path):
    """Extract features from an image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return None  # Return None if image not found

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    flattened = hist.flatten()
    return flattened

def process_dataset(dataset_dir):
    """Process the dataset to extract features and labels."""
    features = []
    labels = []

    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                feature = extract_features(image_path)
                
                if feature is None or feature.size == 0:
                    print(f"Feature extraction failed for image: {image_path}")
                    continue

                features.append(feature)
                labels.append(label)  # Use the folder name as the label

    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    if features.size == 0 or len(features.shape) != 2:
        raise ValueError("No valid features were extracted from the images.")
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Apply scaling
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    
    # Apply PCA
    n_components = min(features.shape[0], features.shape[1])  # Set n_components to the min of (n_samples, n_features)
    pca = PCA(n_components=n_components)  # Adjust as needed
    features = pca.fit_transform(features)
    
    return features, labels, scaler, pca

def evaluate_model(features, labels):
    """Evaluate the model and calculate performance metrics using Leave-One-Out Cross-Validation."""
    loo = LeaveOneOut()
    model = KNeighborsClassifier(n_neighbors=1)

    all_predictions = []
    all_true_labels = []

    for train_index, test_index in loo.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)

    # Print predictions and true labels for debugging
    print("Predictions:", all_predictions)
    print("True labels:", all_true_labels)

    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)

    return accuracy, precision, recall, conf_matrix

def main():
    dataset_dir = r'E:\PythonApplication2 (2)\PythonApplication2\testing'  # Update this path
    output_dir = r'E:\PythonApplication2 (2)\PythonApplication2\output'
    
    os.makedirs(output_dir, exist_ok=True)

    features, labels, scaler, pca = process_dataset(dataset_dir)
    
    # Save the model
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(output_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    # Evaluate the model
    accuracy, precision, recall, conf_matrix = evaluate_model(features, labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

if __name__ == "__main__":
    main()

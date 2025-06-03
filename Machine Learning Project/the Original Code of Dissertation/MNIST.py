import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values.astype(np.float32)
    return images, labels

# Load csv Data
train_images, train_labels = load_csv_data(r'C:\Users\Huawei\Desktop\fashion-mnist_train.csv')
test_images, test_labels = load_csv_data(r'C:\Users\Huawei\Desktop\fashion-mnist_test.csv')

# Data Preprocessing
def preprocess_data(images, labels):
    # Normalization
    images = images / 255.0
    # Convert to 28x28 Image Format
    images = images.reshape(-1, 28 * 28)
    return images, labels

X_train, y_train = preprocess_data(train_images, train_labels)
X_test, y_test = preprocess_data(test_images, test_labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Visualization Comparison
methods = {
    "PCA 2D": PCA(n_components=2),
    "t-SNE 2D": TSNE(n_components=2, random_state=42),
    "UMAP 2D": umap.UMAP(n_components=2, random_state=42),
    "PCA 3D": PCA(n_components=3),
    "t-SNE 3D": TSNE(n_components=3, random_state=42),
    "UMAP 3D": umap.UMAP(n_components=3, random_state=42)
}

plt.figure(figsize=(24, 12)) 

for i, (name, model) in enumerate(methods.items()):
    row = 0 if i < 3 else 1  
    col = i % 3
    # Dimensionality Reduction
    X_red = model.fit_transform(X_train_scaled)
    ax = plt.subplot(2, 3, row*3 + col + 1, projection='3d' if '3D' in name else None)
    # 3D Plot
    if '3D' in name:
        sc = ax.scatter(X_red[:,0], X_red[:,1], X_red[:,2], 
                       c=y_train, cmap='Spectral', alpha=0.6, s=2)
        ax.set_zlabel('Component 3')
    # 2D Plot
    else:
        sc = ax.scatter(X_red[:,0], X_red[:,1], 
                       c=y_train, cmap='Spectral', alpha=0.6, s=2)
    
    ax.set_title(f"{name} (n={len(X_train)})")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.colorbar(sc, ax=ax, label='Class')
plt.tight_layout()
plt.show()

#Classification Performance Validation
dim_reducers = {
    "Original": None,
    "PCA 2D": PCA(n_components=2),
    "PCA 3D": PCA(n_components=3),
    "UMAP 2D": umap.UMAP(n_components=2, random_state=42),
    "UMAP 3D": umap.UMAP(n_components=3, random_state=42)
}

results = []
for name, model in dim_reducers.items():
    print(f"\nProcessing {name}...")
    
    # Dimensionality Reduction
    if model:
        X_train_red = model.fit_transform(X_train_scaled)
        X_test_red = model.transform(X_test_scaled)
    else:
        X_train_red = X_train_scaled
        X_test_red = X_test_scaled
    
    #Use SVM with RBF Kernel
    svm = SVC(kernel='rbf', random_state=42, verbose=1)
    svm.fit(X_train_red, y_train)
    
    # Predict and Evaluate
    y_pred = svm.predict(X_test_red)
    
    # Calculate Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1": f1_score(y_test, y_pred, average='macro')
    }
    
    results.append((name, metrics))
    
    # Output Detailed Classification Report
    print(f"\n{name} classification report:")
    print(classification_report(y_test, y_pred, digits=4))

print("\nClassification Performance Comparison :")
print("{:<10} | {:<8} | {:<8} | {:<8} | {:<8}".format(
    "Method", "Acc", "Precision", "Recall", "F1"))
print("-" * 60)
for res in results:
    name, metrics = res
    print("{:<10} | {:.4f}  | {:.4f}    | {:.4f}   | {:.4f}".format(
        name,
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1"]))

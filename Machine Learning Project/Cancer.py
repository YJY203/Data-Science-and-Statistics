import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

# Load the Breast Cancer Dataset
cancer_data = datasets.load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
target = cancer_data.target

# Standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=10) 
selected_features = selector.fit_transform(scaled_features, target)

X = selected_features
y = cancer_data.target

# Dimensionality Reduction Visualization
def plot_embeddings(X, y, title, dim=2):
    plt.figure(figsize=(15, 5))
    
    # Define Dimensionality Reduction Methods
    methods = {
        "PCA": PCA(n_components=dim),
        "t-SNE": TSNE(n_components=dim, random_state=42),
        "UMAP": umap.UMAP(n_components=dim, random_state=42)
    }
    
    for i, (name, model) in enumerate(methods.items()):
        try:
            if dim == 3:
                ax = plt.subplot(1, 3, i+1, projection='3d')
                emb = model.fit_transform(X)
                ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=y, cmap='viridis', s=20)
                ax.set_zlabel('Component 3')
            else:
                plt.subplot(1, 3, i+1)
                emb = model.fit_transform(X)
                plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap='viridis', s=20)
            
            plt.title(f"{name} {dim}D")
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
        except Exception as e:
            print(f"{name} {dim}D 降维失败: {str(e)}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot 2D and 3D Visualizations
plot_embeddings(X, y, "Feature Selected Data (10 features)", dim=2)
plot_embeddings(X, y, "Feature Selected Data (10 features)", dim=3)




# PCA + SVM Classification Performance Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Range of Dimensions to Test
max_features = X_train.shape[1]
n_components = list(range(4, max_features+1))
accuracies = []

# Test Different PCA Dimensions
for n in n_components:
    # PCA Dimensionality Reduction
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # SVM Classification
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"n_components={n}, Accuracy={acc:.4f}")

# Original Data (No Dimensionality Reduction) Accuracy
svm_orig = SVC(kernel='linear', random_state=42)
svm_orig.fit(X_train, y_train)
y_pred_orig = svm_orig.predict(X_test)
acc_orig = accuracy_score(y_test, y_pred_orig)
print(f"\nOriginal data accuracy: {acc_orig:.4f}")

# Plot Accuracy Changes
plt.figure(figsize=(10, 6))
plt.plot(n_components, accuracies, marker='o', linestyle='--', color='b')
plt.axhline(y=acc_orig, color='r', linestyle='--', label='Original Data')
plt.xlabel("Number of PCA Components")
plt.ylabel("Classification Accuracy")
plt.title("PCA Dimension Reduction vs SVM Accuracy")
plt.xticks(n_components)
plt.grid(True)
plt.legend()
plt.show()



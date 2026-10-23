import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = load_digits()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

K_values = range(2, 16)
inertias = []
silhouette_scores = []

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print("K =", k)
    print("Inertia:", round(kmeans.inertia_, 2))
    print("Silhouette Score:", round(score, 4))
    print()

plt.figure(figsize=(8, 5))
plt.plot(K_values, inertias, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Curve")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(K_values, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.grid()
plt.show()

best_index = silhouette_scores.index(max(silhouette_scores))
best_k = list(K_values)[best_index]

print("Best K based on Silhouette Score:", best_k)
print("Highest Silhouette Score:", round(max(silhouette_scores), 4))

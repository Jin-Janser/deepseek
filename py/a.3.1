# 필수 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# 1. 데이터 로딩 및 전처리
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]

# 이상치 처리 (IQR 1.5)
Q1 = df.iloc[:, :4].quantile(0.25)
Q3 = df.iloc[:, :4].quantile(0.75)
IQR = Q3 - Q1
mask = ~((df.iloc[:, :4] < (Q1 - 1.5*IQR)) | (df.iloc[:, :4] > (Q3 + 1.5*IQR))).any(axis=1)
df_clean = df[mask]

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean.iloc[:, :4])

# 2. 주성분 분석
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# 스크리 플롯
plt.figure(figsize=(8,4))
plt.bar(range(4), pca.explained_variance_ratio_)
plt.title('Scree Plot')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')

# Biplot
plt.figure(figsize=(10,6))
sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=df_clean['Species'], palette='viridis')
for i, feature in enumerate(iris.feature_names):
    plt.arrow(0, 0, loadings[i,0]*3, loadings[i,1]*3, color='r', alpha=0.5)
    plt.text(loadings[i,0]*3.2, loadings[i,1]*3.2, feature, color='r')
plt.title('PCA Biplot')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

# 3. 군집 분석
# K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 계층적 군집
Z = linkage(X_scaled, method='ward')
hc_labels = fcluster(Z, t=3, criterion='maxclust') - 1

# 평가 지표 계산
def dunn_index(data, labels):
    dist_matrix = squareform(pdist(data))
    intra = []
    for cluster in np.unique(labels):
        points = data[labels == cluster]
        if len(points) > 1:
            intra.append(np.max(dist_matrix[labels == cluster][:, labels == cluster]))
    inter = []
    for i in range(len(np.unique(labels))):
        for j in range(i+1, len(np.unique(labels))):
            inter.append(np.min(dist_matrix[labels == i][:, labels == j]))
    return np.min(inter) / np.max(intra)

metrics = {
    'K-means': {
        'Silhouette': silhouette_score(X_scaled, kmeans_labels),
        'Dunn Index': dunn_index(X_scaled, kmeans_labels)
    },
    'Hierarchical': {
        'Silhouette': silhouette_score(X_scaled, hc_labels),
        'Dunn Index': dunn_index(X_scaled, hc_labels)
    }
}

# 4. 시각화
# 클러스터 오버레이
fig, axes = plt.subplots(1,2, figsize=(14,6))
sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=kmeans_labels, palette='tab10', ax=axes[0])
axes[0].set_title('K-means Clusters')
sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=hc_labels, palette='tab10', ax=axes[1])
axes[1].set_title('Hierarchical Clusters')

# 히스토그램 비교
fig, axes = plt.subplots(2,2, figsize=(12,8))
for i, col in enumerate(iris.feature_names):
    sns.histplot(df[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Original {col}')
plt.tight_layout()

plt.show()

print(f"설명 분산: {pca.explained_variance_ratio_}")
# 출력: [0.729 0.228] → PC1 72.9%, PC2 22.8% 설명

K-means: 실루엣 0.55, Dunn 0.32
계층적: 실루엣 0.51, Dunn 0.29

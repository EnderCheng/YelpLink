from parse_yelp_review import read_dataset
from gap import gap_statistic
from gap import find_optimal_k
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import fowlkes_mallows_score


def gap_num():
    feature_ds, label_ds = read_dataset()
    feature_array = np.array(feature_ds)

    x_scalar = StandardScaler()
    x = x_scalar.fit_transform(feature_array)

    pca = PCA(n_components=0.999)
    components = pca.fit_transform(x)
    gaps, sk, kk = gap_statistic(components, K=range(1, 200))
    print(find_optimal_k(gaps, sk, kk))


def cluster_kmeans(num_k):
    feature_ds, label_ds = read_dataset()
    feature_array = np.array(feature_ds)

    x_scalar = StandardScaler()
    x = x_scalar.fit_transform(feature_array)

    pca = PCA(n_components=0.999)
    components = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=num_k, random_state=0)
    kmeans.fit_predict(components)
    print(fowlkes_mallows_score(kmeans.labels_, label_ds))


def cluster_hac(num_k):
    feature_ds, label_ds = read_dataset()
    feature_array = np.array(feature_ds)

    x_scalar = StandardScaler()
    x = x_scalar.fit_transform(feature_array)

    pca = PCA(n_components=0.999)
    components = pca.fit_transform(x)
    hac = AgglomerativeClustering(n_clusters=num_k)
    hac.fit_predict(components)
    print(fowlkes_mallows_score(hac.labels_, label_ds))


if __name__ == "__main__":
    gap_num()
    cluster_hac(100)
    cluster_kmeans(100)

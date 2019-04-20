from parse_yelp_review import read_dataset
from gap import gap_statistic
from gap import find_optimal_k
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import fowlkes_mallows_score


def gap_num(num_user):
    feature_ds, label_ds = read_dataset()

    user_max_id = num_user - 1
    sub_feature_ds = []
    for i in range(0, len(label_ds)):
        if label_ds[i] <= user_max_id:
            sub_feature_ds.append(feature_ds[i])

    feature_array = np.array(sub_feature_ds)

    x_scalar = StandardScaler()
    x = x_scalar.fit_transform(feature_array)

    pca = PCA(n_components=0.999)
    components = pca.fit_transform(x)
    gaps, sk, kk = gap_statistic(components, K=range(1, num_user))
    print("gap_statistic")
    opt_k = find_optimal_k(gaps, sk, kk)
    print(opt_k)
    print(num_user)
    print(abs(opt_k - num_user))


def cluster_kmeans(num_k):
    feature_ds, label_ds = read_dataset()

    user_max_id = num_k - 1
    sub_feature_ds = []
    sub_label_ds = []
    for i in range(0, len(label_ds)):
        if label_ds[i] <= user_max_id:
            sub_feature_ds.append(feature_ds[i])
            sub_label_ds.append(label_ds[i])

    feature_array = np.array(sub_feature_ds)

    x_scalar = StandardScaler()
    x = x_scalar.fit_transform(feature_array)

    pca = PCA(n_components=0.999)
    components = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=num_k, random_state=0)
    kmeans.fit_predict(components)
    print("kmeans")
    print(fowlkes_mallows_score(kmeans.labels_, sub_label_ds))


def cluster_hac(num_k):
    feature_ds, label_ds = read_dataset()

    user_max_id = num_k - 1
    sub_feature_ds = []
    sub_label_ds = []
    for i in range(0, len(label_ds)):
        if label_ds[i] <= user_max_id:
            sub_feature_ds.append(feature_ds[i])
            sub_label_ds.append(label_ds[i])

    feature_array = np.array(sub_feature_ds)

    x_scalar = StandardScaler()
    x = x_scalar.fit_transform(feature_array)

    pca = PCA(n_components=0.999)
    components = pca.fit_transform(x)
    hac = AgglomerativeClustering(n_clusters=num_k, linkage='average')
    hac.fit_predict(components)
    print("HAC")
    print(fowlkes_mallows_score(hac.labels_, sub_label_ds))


if __name__ == "__main__":
    for user_n in range(2, 10):
        gap_num(user_n)

    for user_n in range(10, 110, 10):
        cluster_hac(user_n)
        cluster_kmeans(user_n)

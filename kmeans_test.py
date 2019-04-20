from parse_yelp_review import read_dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import fowlkes_mallows_score


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
    print(fowlkes_mallows_score(kmeans.labels_, sub_label_ds))


if __name__ == "__main__":
    for user_n in range(10, 110, 10):
        cluster_kmeans(user_n)

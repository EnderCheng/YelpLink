from parse_yelp_review import get_feature_dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def classification_randomforest(num_t):
    train_dataset, train_label, test_dataset, test_label = get_feature_dataset(num_t)

    x = np.array(train_dataset)
    y = np.array(train_label)
    t = np.array(test_dataset)
    real = np.array(test_label)

    print(x.shape)
    print(t.shape)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    t = scaler.transform(t)

    pca = PCA(n_components=0.999)
    x_components = pca.fit_transform(x)
    t_components = pca.transform(t)

    rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=2)
    rfc.fit(x_components, y)
    ret = rfc.predict(t_components)
    accuracy = rfc.score(t_components, real)
    print(accuracy)


if __name__ == "__main__":
    for user_n in range(10, 110, 10):
        classification_randomforest(user_n)

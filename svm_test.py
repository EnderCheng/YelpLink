from parse_yelp_review import get_feature_dataset
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def classification_svm(num_t):
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

    print("scaler finish")
    pca = PCA(n_components=0.999)
    x_components = pca.fit_transform(x)
    t_components = pca.transform(t)
    print("pca finish")

    dual = True
    if x_components.shape[0] > x_components.shape[1]:
        dual = False

    lsvc = LinearSVC(random_state=0, dual=dual)
    lsvc.fit(x_components, y)
    ret = lsvc.predict(t_components)
    accuracy = lsvc.score(t_components, real)
    print("svm")
    print(accuracy)


if __name__ == "__main__":
    for user_n in range(10, 110, 10):
        print(user_n)
        classification_svm(user_n)

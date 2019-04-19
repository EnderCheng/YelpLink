from parse_yelp_review import get_feature_dataset
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def classification_nb(num_t):
    train_dataset, train_label, test_dataset, test_label = get_feature_dataset(num_t)

    x = np.array(train_dataset)
    y = np.array(train_label)
    t = np.array(test_dataset)
    real = np.array(test_label)

    gnb = GaussianNB()
    gnb.fit(x, y)
    ret = gnb.predict(t)
    accuracy = gnb.score(t, real)
    print(accuracy)


def classification_svm(num_t):
    train_dataset, train_label, test_dataset, test_label = get_feature_dataset(num_t)

    x = np.array(train_dataset)
    y = np.array(train_label)
    t = np.array(test_dataset)
    real = np.array(test_label)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    t = scaler.transform(t)
    print('Scaling finish')

    pca = PCA(n_components=0.999)
    x_components = pca.fit_transform(x)
    t_components = pca.transform(t)
    print('PCA finish')

    dual = True
    if x_components.shape[0] > x_components.shape[1]:
        dual = False

    lsvc = LinearSVC(random_state=0, dual=dual)
    lsvc.fit(x_components, y)
    ret = lsvc.predict(t_components)
    accuracy = lsvc.score(t_components, real)
    print(accuracy)


def classification_randomforest(num_t):
    train_dataset, train_label, test_dataset, test_label = get_feature_dataset(num_t)

    x = np.array(train_dataset)
    y = np.array(train_label)
    t = np.array(test_dataset)
    real = np.array(test_label)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    t = scaler.transform(t)
    print('Scaling finish')

    pca = PCA(n_components=0.999)
    x_components = pca.fit_transform(x)
    t_components = pca.transform(t)
    print('PCA finish')

    rfc = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rfc.fit(x_components, y)
    ret = rfc.predict(t_components)
    accuracy = rfc.score(t_components, real)
    print(accuracy)


if __name__ == "__main__":
    # classification_nb(50)
    # classification_svm(50)
    classification_randomforest(50)



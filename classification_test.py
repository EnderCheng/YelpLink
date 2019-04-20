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

    print(x.shape)
    print(t.shape)

    gnb = GaussianNB()
    print("start fit")
    gnb.fit(x, y)
    print("end fit")
    print("start predict")
    ret = gnb.predict(t)
    print("end predict")
    accuracy = gnb.score(t, real)
    print("naive bayes")
    print(accuracy)


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


def classification_randomforest(num_t):
    train_dataset, train_label, test_dataset, test_label = get_feature_dataset(num_t)

    x = np.array(train_dataset)
    y = np.array(train_label)
    t = np.array(test_dataset)
    real = np.array(test_label)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    t = scaler.transform(t)

    pca = PCA(n_components=0.999)
    x_components = pca.fit_transform(x)
    t_components = pca.transform(t)

    rfc = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rfc.fit(x_components, y)
    ret = rfc.predict(t_components)
    accuracy = rfc.score(t_components, real)
    print("random forest")
    print(accuracy)


if __name__ == "__main__":
    for user_n in range(10, 110, 10):
        print(user_n)
        classification_nb(user_n)
        # classification_svm(user_n)
        # classification_randomforest(user_n)




from parse_yelp_review import get_feature_dataset
import numpy as np
from sklearn.naive_bayes import GaussianNB


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


if __name__ == "__main__":
    for user_n in range(10, 110, 10):
        print(user_n)
        classification_nb(user_n)




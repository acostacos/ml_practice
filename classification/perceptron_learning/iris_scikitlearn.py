import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Load training data
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))

    # Split training data into test data and training data
    # 30% test data, 70% training data
    # Stratify - get same proportion of test data labels from input data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y,
    )

    # Feature scaling
    sc = StandardScaler()
    # Estimates mean and standard deviation of training data
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Train Data
    ppn = Perceptron(max_iter=40, eta0=0.01, random_state=1)
    ppn.fit(X_train_std, y_train)

    # Predict
    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    # Measure metrics
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    # print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

if __name__ == "__main__":
    main()
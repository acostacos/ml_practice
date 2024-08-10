import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from Perceptron import Perceptron

def main():
    FILENAME = "iris.data"

    # Read data
    df = pd.read_csv(f"../../data/{FILENAME}", header=None)
    # print(df.tail())

    # Select setosa and versicolor
    y = df.iloc[0:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)

    # Extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # Plot data
    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.xlabel('sepal length (cm)')
    # plt.ylabel('petal length (cm)')
    # plt.legend(loc='upper left')
    # plt.show()

    # Train model using perceptron learning algorithm
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    # Plot errors from epochs
    # plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of updates')
    # plt.show()

    # Plot decision regions with data points
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


def plot_decision_regions(X: np.ndarray, y: np.ndarray, classifier, resolution: float = 0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot case samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from AdalineGD import AdalineGD
from AdalineSGD import AdalineSGD
from matplotlib.colors import ListedColormap


def main():
    FILENAME = "iris.data"

    # Read data
    df = pd.read_csv(f"../../data/{FILENAME}", header=None)

    # Select setosa and versicolor
    y = df.iloc[0:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)

    # Extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # # Plot for comparing training based on the learning rate
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # # Train model using Adaline learning algorithm
    # # Learning rate that is too big
    # ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    # ax[0].plot(range(1, len(ada1.cost) + 1),
    #            np.log10(ada1.cost), marker='o')
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('log(Sum-squared-error)')
    # ax[0].set_title('Adaline - Learning rate 0.01')

    # # Learning rate that is too small
    # ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    # ax[1].plot(range(1, len(ada2.cost) + 1),
    #            ada2.cost, marker='o')
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_ylabel('Sum-squared-error')
    # ax[1].set_title('Adaline - Learning rate 0.0001')

    # plt.show()

    # Feature Scaling
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:,1] = (X[:,1] - X[:, 1].mean()) / X[:, 1].std()

    # # Train model with big learning after Feature Scaling
    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada.cost) + 1), ada.cost, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

    # Adaline with Stochastic Gradient Descent
    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
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
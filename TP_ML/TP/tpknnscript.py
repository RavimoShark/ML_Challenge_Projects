"""Created some day.

@authors:  salmon, gramfort, vernade
"""
from functools import partial  # useful for weighted distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
from scipy import stats  # to use scipy.stats.mode
from sklearn import neighbors
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tp_knn_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                           rand_checkers, rand_clown, plot_2d, ErrorCurve,
                           frontiere_new, LOOCurve)


import seaborn as sns
from matplotlib import rc

plt.close('all')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 12,
          'font.size': 16,
          'legend.fontsize': 16,
          'text.usetex': False,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
sns.axes_style()

############################################################################
#     Data Generation: example
############################################################################

# Q1

np.random.seed(42)  # fix seed globally

n = 100
mu = [1., 1.]
sigma = [1., 1.]
rand_gauss(n, mu, sigma)

n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
X2, y2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
X3, y3 = rand_clown(n1, n2, sigma1, sigma2)

n1 = 150
n2 = 150
sigma = 0.1
X4, y4 = rand_checkers(n1, n2, sigma)

############################################################################
#     Displaying labeled data
############################################################################

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.subplot(141)
plt.title('First data set')
plot_2d(X1, y1)

plt.subplot(142)
plt.title('Second data set')
plot_2d(X2, y2)

plt.subplot(143)
plt.title('Third data set')
plot_2d(X3, y3)

plt.subplot(144)
plt.title('Fourth data set')
plot_2d(X4, y4)

############################################################################
#     K-NN
############################################################################

# Q2 : Write your own implementation


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """Home made KNN Classifier class."""

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        # TODO: Compute all pairwise distances between X and self.X_ using e.g.
        # metrics.pairwise.pairwise_distances
        dist =
        # Get indices to sort them
        idx_sort =
        # Get indices of neighbors
        idx_neighbors =
        # Get labels of neighbors
        y_neighbors =
        # Find the predicted labels y for each entry in X
        # You can use the scipy.stats.mode function
        mode, _ =
        # the following might be needed for dimensionaality
        #y_pred = np.asarray(mode.ravel(), dtype=np.intp)
        return y_pred

# TODO : compare your implementation with scikit-learn

# Focus on dataset 2 for instance
X_train = X2[::2]
Y_train = y2[::2].astype(int)
X_test = X2[1::2]
Y_test = y2[1::2].astype(int)

# TODO: use KNeighborsClassifier vs. KNNClassifier


# Q3 : test now all datasets
# From now on use the Scikit-Learn implementation

n_neighbors = 5  # the k in k-NN
knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

for X, y in [(X1, y1), (X2, y2), (X3, y3), (X4, y4)]:

    def f(xx):
        """Classifier: needed to avoid warning due to shape issues."""
        return knn.predict(xx.reshape(1, -1))
    knn.fit(X, y)
    plt.figure()
    plot_2d(X, y)
    n_labels = np.unique(y).shape[0]
    frontiere_new(f, X, y, w=None, step=50, alpha_choice=1, n_labels=n_labels,
                  n_neighbors=n_neighbors)


# Q4: Display the result when varying the value of K

plt.figure(3, figsize=(12, 8))
plt.subplot(3, 5, 3)
plot_2d(X_train, Y_train)
plt.xlabel('Samples')
ax = plt.gca()
ax.get_yaxis().set_ticks([])
ax.get_xaxis().set_ticks([])

for n_neighbors in range(1, 11):
    # TODO : fit the knn
    plt.subplot(3, 5, 5 + n_neighbors)
    plt.xlabel('KNN with k=%d' % n_neighbors)

    def f(xx):
        """Classifier: needed to avoid warning due to shape issues."""
        return knn.predict(xx.reshape(1, -1))

    n_labels = np.unique(y).shape[0]
    frontiere_new(f, X, y, w=None, step=50, alpha_choice=1, n_labels=n_labels,
                  colorbar=False, samples=False, n_neighbors=n_neighbors)
    plt.draw()  # update plot

plt.tight_layout()


def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return wknn.predict(xx.reshape(1, -1))


frontiere_new(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

print(wknn.predict(X_train))

# Q5 : Scores on train data

# TODO

# Q6 : Scores on left out data


n1 = n2 = 200
sigma = 0.1
data4 = rand_checkers(2 * n1, 2 * n2, sigma)

X_train = X4[::2]
Y_train = y4[::2].astype(int)
X_test = X4[1::2]
Y_test = y4[1::2].astype(int)

# TODO


############################################################################
#     Digits data
############################################################################

# Q8 : test k-NN on digits dataset


# The digits dataset:
# from sklearn import datasets
digits = datasets.load_digits()

print(type(digits))
# A Bunch is a subclass of 'dict' (dictionary)
# help(dict)
# see also "http://docs.python.org/2/library/stdtypes.html#mapping-types-dict"

plt.close(7)
plt.figure(7)
for index, (img, label) in enumerate(list(zip(digits.images, digits.target))[10:20]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='None')
    plt.title('Training: %i' % label)

plt.figure()
plt.hist(digits.target,normed=True)
plt.title("Digits histogram over the whole dataset")
plt.ylabel("Frequency")


n_samples = len(digits.data)

X_train = digits.data[:n_samples // 2]
Y_train = digits.target[:n_samples // 2]
X_test = digits.data[n_samples // 2:]
Y_test = digits.target[n_samples // 2:]

knn = neighbors.KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, Y_train)

score = knn.score(X_test, Y_test)
print('Score : %s' % score)


# Q9 : Compute confusion matrix

# from sklearn.metrics import confusion_matrix

Y_pred = knn.predict(X_test)

# TODO : compute and show confusion matrix

# Q10 : Estimate k with cross-validation for instance

# Have a look at the class  'LOOCurve', defined in the source file.
# from tp_knn_source import LOOCurve


loo_curve = LOOCurve(k_range=list(range(1, 50, 5)) + list(range(100, 300, 100)))

# TODO


# Q11: Weighted k-NN


def weights(dist, h=0.1):
    """Return array of weights, exponentially small w.r.t. the distance.

    Parameters
    ----------
    dist : a one-dimensional array of distances.

    Returns
    -------
    weight : array of the same size as dist
    """

    return  # TODO


n_neighbors = 5
wknn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                      weights=partial(weights, h=0.01))
wknn.fit(X_train, Y_train)


def f(xx):
    """Classifier: needed to avoid warning due to shape issues."""
    return wknn.predict(xx.reshape(1, -1))

plt.figure(5)
plot_2d(X_train, Y_train)
frontiere_new(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

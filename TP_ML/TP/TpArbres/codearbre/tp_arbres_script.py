#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree, datasets

from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown, plot_2d,
                              frontiere)

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
_ = sns.axes_style()

############################################################################
# Data Generation: example
############################################################################

np.random.seed(1)

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
data1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
data2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
data3 = rand_clown(n1, n2, sigma1, sigma2)


n1 = 114  # XXX : change
n2 = 114
n3 = 114
n4 = 114
sigma = 0.1
data4 = rand_checkers(n1, n2, n3, n4, sigma)

############################################################################
# Displaying labeled data
############################################################################

plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.subplot(141)
plt.title('First data set')
plot_2d(data1[:, :2], data1[:, 2], w=None)

plt.subplot(142)
plt.title('Second data set')
plot_2d(data2[:, :2], data2[:, 2], w=None)

plt.subplot(143)
plt.title('Third data set')
plot_2d(data3[:, :2], data3[:, 2], w=None)

plt.subplot(144)
plt.title('Fourth data set')
plot_2d(data4[:, :2], data4[:, 2], w=None)


############################################
# ARBRES
############################################


# Q2. Créer un objet 'arbre de décision' en spécifiant le critère de
# classification comme l'indice de gini ou l'entropie, avec la
# fonction 'DecisionTreeClassifier' du module 'tree'.

dt = tree.DecisionTreeClassifier()

# Effectuer la classification d'un jeu de données simulées

# data = ...
# X = ...
# Y = ...

dt.fit(X, Y)

# Afficher les scores en fonction du paramètre max_depth

dmax = 12
scores = np.zeros(dmax)
plt.close(2)
plt.figure(2, figsize=(15, 10))

for i in range(dmax):
    # XXX : TODO

print(scores)

plt.close(3)
plt.figure(3)
# plt.plot(...)  # TODO
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')

# Q3 Afficher les frontières obtenues avec l'arbre pour le meilleur paramètre

# TODO
# frontiere_new(lambda x: dt.predict(x.reshape((1, -1))), X, Y, step=100)

# Q4.  Exporter la représentation graphique de l'arbre: Need graphviz installed

# tree.export_graphviz(dt, out_file="myTestTree.dot", filled=True)
# import os
# os.system("dot -Tpdf myTestTree.dot -o myTestTree.pdf")
# !open myTestTree.pdf
# os.system("evince myTestTree.pdf")

# Q5 : Génération de base de test

# # data_test = rand_checkers(...
# X_test = data_test[:, :2]
# Y_test = data_test[:, 2].astype(int)

dmax = 12
scores = np.zeros(dmax)
plt.close(5)
plt.figure(5, figsize=(15, 10))

for i in range(dmax):
    # TODO

plt.close(6)
plt.figure(6)
# plt.plot(...)  # TODO
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
print(scores)

# Q6. même question avec les données de reconnaissances de texte 'digits'

# Import the digits dataset
digits = datasets.load_digits()

n_samples = len(digits.data)
X = digits.data[:n_samples // 2]  # digits.images.reshape((n_samples, -1))
Y = digits.target[:n_samples // 2]
X_test = digits.data[n_samples // 2:]
Y_test = digits.target[n_samples // 2:]

# TODO

# Q7. estimer la meilleur profondeur avec un cross_val_score

# TODO



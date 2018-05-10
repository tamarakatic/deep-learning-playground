import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import pandas as pd
from pylab import bone, pcolor, colorbar, plot, show

from sklearn.preprocessing import MinMaxScaler

from minisom import MiniSom


def discover_frauds():
    dataset = pd.read_csv('credit_card_applications.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)

    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o', 's']
    colors = ['r', 'g']
    for i, x in enumerate(X):
        w = som.winner(x)
        plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor=colors[y[i]],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=2)

    show()

    mappings = som.win_map(X)
    frauds = np.concatenate((mappings[(5, 1)], mappings[(7, 1)]), axis=0)
    return sc.inverse_transform(frauds), dataset
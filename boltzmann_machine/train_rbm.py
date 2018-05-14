import numpy as np
import pandas as pd

from rbm import RestrictedBoltzmanMachines
from preprocessing import read_csv, converts, convert_rating_to_binary

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


movies = read_csv('movie_lens1m/movies.dat')
users = read_csv('movie_lens1m/users.dat')
ratings = read_csv('movie_lens1m/ratings.dat')

training_set = pd.read_csv('movie_lens100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('movie_lens100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

num_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
num_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

training_set = converts(training_set, num_users, num_movies)
test_set = converts(test_set, num_users, num_movies)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

training_set = convert_rating_to_binary(training_set)
test_set = convert_rating_to_binary(test_set)

num_visible_nodes = len(training_set[0])
num_hidden_nodes = 100
batch_size = 100

rbm = RestrictedBoltzmanMachines(num_visible_nodes, num_hidden_nodes)

# Train the RBM
num_epoch = 10
for epoch in range(1, num_epoch + 1):
    train_loss = 0
    counter = 0.
    for id_user in range(0, num_users - batch_size, batch_size):
        visible_k = training_set[id_user:id_user + batch_size]
        visible_0 = training_set[id_user:id_user + batch_size]
        prob_hidden_0, _ = rbm.sample_hidden_nodes(visible_0)
        for k in range(10):
            _, hidden_k = rbm.sample_hidden_nodes(visible_k)
            _, visible_k = rbm.sample_visible_nodes(hidden_k)
            visible_k[visible_0 < 0] = visible_0[visible_0 < 0]
        prob_hidden_k, _ = rbm.sample_hidden_nodes(visible_k)
        rbm.train(visible_0, visible_k, prob_hidden_0, prob_hidden_k)
        train_loss += torch.mean(torch.abs(visible_0[visible_0 >= 0] - visible_k[visible_0 >= 0]))  # Average distance
        counter += 1.
    print("epoch: " + str(epoch) + ' loss: ' + str(train_loss / counter))

import numpy as np
import pandas as pd


def read_csv(path):
    return pd.read_csv('{}'.format(path), sep='::', header=None, engine='python',encoding='latin-1')


def converts(data, num_users, num_movies):
    new_data = []
    for id_users in range(1, num_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(num_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


# Converting the rating into binary ratings 1 as Liked and 0 as Not Liked
def convert_rating_to_binary(type_of_set):
    type_of_set[type_of_set == 0] = -1
    type_of_set[type_of_set == 1] = 0
    type_of_set[type_of_set == 2] = 0
    type_of_set[type_of_set >= 3] = 1
    return type_of_set

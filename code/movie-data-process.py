import numpy as np
import pandas as pd
from fancyimpute import SoftImpute


# -----------------------------------------------------------
# read MovieTweetings data via
# https://github.com/sidooms/MovieTweetings/tree/master/snapshots/200K
# and process data
# -----------------------------------------------------------


url = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/snapshots/200K/"

users = pd.read_csv(url + "users.dat", header=None, sep='::', engine='python')
movies = pd.read_csv(url + 'movies.dat', header=None, sep='::', engine='python')
ratings = pd.read_csv(url + 'ratings.dat', header=None, sep='::', engine='python')

ratings = ratings.rename(columns={0: "user", 1: "movie", 2: "rate", 3: "timestamp"})

freq_users = ratings.user.value_counts()
freq_movies = ratings.movie.value_counts()

num_users = sum(freq_users >= 50)
num_movies = sum(freq_movies >= 50)
num_ratings = len(ratings)
users_ = [i for i in users[0] if freq_users[i] >= 50]
users_index = {users_[i]: i for i in range(len(users_))}
movies_ = [j for j in movies[0] if freq_movies[j] >= 50]
movies_index = {movies_[j]: j for j in range(len(movies_))}

v_matrix = np.zeros(shape=(num_users, num_movies))
v_matrix[:] = np.nan
for i in range(num_ratings):
    if ratings.user[i] in users_ and ratings.movie[i] in movies_:
        x = users_index[ratings.user[i]]
        y = movies_index[ratings.movie[i]]
        v_matrix[x, y] = ratings.rate[i]

# ---
# do matrix completion by iterative soft thresholding of SVD decompositions with fancyimpute
# ---

X_filled_softimpute = SoftImpute().fit_transform(v_matrix)
np.save('../data/movie_rating.npy', np.round(np.minimum(np.maximum(X_filled_softimpute, 0), 10), 3))

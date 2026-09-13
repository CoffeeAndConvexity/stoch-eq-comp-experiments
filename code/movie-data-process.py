import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fancyimpute import SoftImpute


# -----------------------------------------------------------
# read MovieTweetings data via
# https://github.com/sidooms/MovieTweetings/tree/master/snapshots/200K
# and process data
# -----------------------------------------------------------


DEFAULT_URL = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/snapshots/200K/"
REPO_ROOT = Path(__file__).resolve().parent.parent


def process_movie_data(base_url=DEFAULT_URL, output=REPO_ROOT / 'data' / 'movie_rating.npy', min_ratings=50):
    users = pd.read_csv(base_url + "users.dat", header=None, sep='::', engine='python')
    movies = pd.read_csv(base_url + 'movies.dat', header=None, sep='::', engine='python')
    ratings = pd.read_csv(base_url + 'ratings.dat', header=None, sep='::', engine='python')
    ratings = ratings.rename(columns={0: "user", 1: "movie", 2: "rate", 3: "timestamp"})

    freq_users = ratings.user.value_counts()
    freq_movies = ratings.movie.value_counts()
    kept_users = [i for i in users[0] if freq_users.get(i, 0) >= min_ratings]
    kept_movies = [j for j in movies[0] if freq_movies.get(j, 0) >= min_ratings]
    users_index = {user: index for index, user in enumerate(kept_users)}
    movies_index = {movie: index for index, movie in enumerate(kept_movies)}

    v_matrix = np.full((len(kept_users), len(kept_movies)), np.nan)
    for rating in ratings.itertuples(index=False):
        x = users_index.get(rating.user)
        y = movies_index.get(rating.movie)
        if x is not None and y is not None:
            v_matrix[x, y] = rating.rate

    if np.any(np.all(np.isnan(v_matrix), axis=1)) or np.any(np.all(np.isnan(v_matrix), axis=0)):
        raise ValueError('Filtering produced a buyer or movie with no observed ratings.')

    # Complete the matrix by iterative soft-thresholded SVD, then retain the
    # rating scale used by the original experiment.
    filled = SoftImpute().fit_transform(v_matrix)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = np.round(np.clip(filled, 0, 10), 3)
    np.save(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and preprocess MovieTweetings 200K.')
    parser.add_argument('--base-url', default=DEFAULT_URL)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'data' / 'movie_rating.npy')
    parser.add_argument('--min-ratings', type=int, default=50)
    args = parser.parse_args()
    process_movie_data(args.base_url, args.output, args.min_ratings)

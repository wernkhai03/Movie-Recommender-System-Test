"""
Download and prepare the MovieLens dataset for the Movie Recommender System.
Uses MovieLens Latest Small (ml-latest-small) for quick development.
"""
import urllib.request
import zipfile
import os
import pandas as pd
import shutil

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_NAME = "ml-latest-small.zip"
EXTRACT_DIR = "ml-latest-small"

def download_and_prepare():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(script_dir, ZIP_NAME)
    extract_path = os.path.join(script_dir, EXTRACT_DIR)

    # Download
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens dataset from {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("Download complete.")
    else:
        print("ZIP already exists, skipping download.")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(script_dir)

    # Prepare movies.csv
    src_movies = os.path.join(extract_path, "movies.csv")
    dst_movies = os.path.join(script_dir, "movies.csv")
    movies = pd.read_csv(src_movies)
    # Rename movieId -> movie_id for consistency
    movies.rename(columns={'movieId': 'movie_id'}, inplace=True)
    # Add an 'id' column (same as movie_id) for compatibility with the app
    movies['id'] = movies['movie_id']
    # Extract year from title  e.g. "Toy Story (1995)" -> 1995
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    # Clean title (remove year part)
    movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    # Compute average_rating and ratings_count from ratings
    src_ratings = os.path.join(extract_path, "ratings.csv")
    ratings = pd.read_csv(src_ratings)
    ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
    
    agg = ratings.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
    agg.columns = ['movie_id', 'average_rating', 'ratings_count']
    movies = movies.merge(agg, on='movie_id', how='left')
    movies['average_rating'] = movies['average_rating'].fillna(0.0)
    movies['ratings_count'] = movies['ratings_count'].fillna(0).astype(int)
    
    movies.to_csv(dst_movies, index=False)
    print(f"Saved {dst_movies}  ({len(movies)} movies)")

    # Prepare ratings.csv
    dst_ratings = os.path.join(script_dir, "ratings.csv")
    ratings = ratings[['user_id', 'movie_id', 'rating']]
    ratings.to_csv(dst_ratings, index=False)
    print(f"Saved {dst_ratings}  ({len(ratings)} ratings)")

    # Cleanup
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("Cleanup complete. Dataset is ready!")

if __name__ == "__main__":
    download_and_prepare()

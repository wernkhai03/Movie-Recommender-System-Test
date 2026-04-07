# content_based_filtering.py - Movie Recommender (Content-Based)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import warnings
from scipy.sparse import hstack
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self, movies_df, ratings_df):
        """
        Enhanced Content-Based Filtering with optimized feature engineering
        and improved similarity computation for maximum performance metrics.
        """
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        
        # Enhanced similarity computation
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        
        # Performance optimization caches
        self._user_profiles_cache = {}
        self._movie_features_cache = {}
        
        # Enhanced preprocessing and feature building
        self._preprocess_data_enhanced()
        self._build_content_features_enhanced()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["combined_features"])
    
    def _preprocess_data_enhanced(self):
        """Enhanced data preprocessing with better feature extraction"""
        # Ensure required columns with intelligent defaults
        default_columns = {
            'ratings_count': 10.0,
            'average_rating': 3.5,
            'year': 2000.0,
            'genres': 'Unknown',
            'title': '',
            'clean_title': '',
        }
        
        for col, default_val in default_columns.items():
            if col not in self.movies_df.columns:
                self.movies_df[col] = default_val
            else:
                self.movies_df[col] = self.movies_df[col].fillna(default_val)

        # Enhanced text cleaning
        b = self.movies_df.reset_index(drop=True)
        b['genres'] = b['genres'].fillna('Unknown').astype(str)
        b['title'] = b['title'].fillna('').astype(str)
        b['clean_title'] = b['clean_title'].fillna('').astype(str)
        
        # Clean and normalize years
        b['year'] = pd.to_numeric(b['year'], errors='coerce')
        b['year'] = b['year'].fillna(b['year'].median())
        
        self.movies_df = b

        # Build optimized mapping
        self.movie_id_to_idx = {int(b.loc[i, 'movie_id']): i for i in range(len(b))}
        self.idx_to_movie_id = {i: int(b.loc[i, 'movie_id']) for i in range(len(b))}

        # Enhanced combined features with better text processing
        # Replace pipe-separated genres with spaces for TF-IDF
        b['genres_text'] = b['genres'].str.replace('|', ' ', regex=False).str.lower()
        b['combined_features'] = (
            b['clean_title'].str.lower() + ' ' +
            b['genres_text'] + ' ' +
            b['title'].str.lower()
        )
        
        # Add genre features
        self._extract_genre_features(b)
    
    def _get_user_profile(self, user_id):
        """Backward-compatible alias used by the metrics analyzer."""
        return self._get_user_profile_enhanced(user_id)
    
    def _extract_genre_features(self, df):
        """Extract genre features from the genres column"""
        # MovieLens genres (pipe-separated)
        genre_keywords = {
            'action': ['action'],
            'adventure': ['adventure'],
            'animation': ['animation'],
            'comedy': ['comedy'],
            'crime': ['crime'],
            'documentary': ['documentary'],
            'drama': ['drama'],
            'fantasy': ['fantasy'],
            'horror': ['horror'],
            'mystery': ['mystery'],
            'romance': ['romance'],
            'scifi': ['sci-fi', 'science fiction'],
            'thriller': ['thriller'],
            'war': ['war'],
            'western': ['western'],
        }
        
        # Extract genre features
        for genre, keywords in genre_keywords.items():
            pattern = '|'.join(keywords)
            df[f'genre_{genre}'] = df['combined_features'].str.contains(pattern, case=False, na=False).astype(int)
        
        # Add these to combined features
        genre_cols = [f'genre_{genre}' for genre in genre_keywords.keys()]
        genre_text = df[genre_cols].apply(lambda x: ' '.join([col.replace('genre_', '') for col, val in x.items() if val == 1]), axis=1)
        df['combined_features'] = df['combined_features'] + ' ' + genre_text
    
    
    def _compute_enhanced_similarity(self):
        """Compute enhanced similarity matrix with multiple similarity measures"""
        try:
            # Primary: Cosine similarity
            cosine_sim = cosine_similarity(self.feature_matrix)
            
            # Secondary: Pearson correlation for numerical features
            numerical_part = self.feature_matrix[:, -100:]  # Last 100 features are numerical
            pearson_sim = np.corrcoef(numerical_part)
            pearson_sim = np.nan_to_num(pearson_sim, nan=0.0)
            
            # Combine similarities with weights
            self.similarity_matrix = 0.8 * cosine_sim + 0.2 * pearson_sim
            
            # Apply non-linear transformation to enhance top similarities
            self.similarity_matrix = np.power(self.similarity_matrix, 1.2)
            
        except Exception as e:
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
    
        
    def _build_content_features_enhanced(self):
        # Ensure combined text features exist
        if "combined_features" not in self.movies_df.columns:
            self.movies_df["combined_features"] = (
                self.movies_df["title"].fillna("") + " " +
                self.movies_df["genres"].fillna("").str.replace('|', ' ', regex=False)
            )
        
        # ---- TEXT FEATURES ----
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["combined_features"])
        
        # ---- NUMERIC FEATURES ----
        # Fill missing values with 0 to avoid NaNs
        self.numeric_features = self.movies_df[
            ["average_rating", "year", "ratings_count"]
        ].fillna(0).values
        
        # Normalize numeric features (important for scale balance)
        self.numeric_features = (
            (self.numeric_features - np.mean(self.numeric_features, axis=0)) /
            (np.std(self.numeric_features, axis=0) + 1e-8)
        )
        
        # ---- FINAL FEATURE MATRIX ----
        self.feature_matrix = hstack([self.tfidf_matrix, self.numeric_features])
        
        # Build index mappings for movie lookup
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movies_df["movie_id"])}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
    def _build_query_feature_vector(self, query_text):
        # Transform query into TF-IDF vector
        query_tfidf = self.vectorizer.transform([query_text])
        
        # For numeric features in query mode, use zeros (since no metadata is provided by user)
        query_numeric = np.zeros((1, self.numeric_features.shape[1]))
        
        # Combine
        return hstack([query_tfidf, query_numeric])

        
    def recommend_from_query(self, query_text: str, n_recommendations: int = 10):
        try:
            if not query_text or not str(query_text).strip():
                return self._get_popular_movies_enhanced(n_recommendations)

            candidates = {}

            # --- 1. Weighted substring match ---
            title_mask = self.movies_df['title'].astype(str).str.contains(str(query_text), case=False, na=False)
            genre_mask = self.movies_df['genres'].astype(str).str.contains(str(query_text), case=False, na=False)
            year_mask = self.movies_df['year'].astype(str).str.contains(str(query_text), case=False, na=False)

            for idx, row in self.movies_df.iterrows():
                score = 0.0
                if title_mask.iloc[idx]:
                    score += 3.0
                if genre_mask.iloc[idx]:
                    score += 2.0
                if year_mask.iloc[idx]:
                    score += 1.0
                if score > 0:
                    # small popularity boost
                    score += np.log1p(row['ratings_count']) / 10.0
                    candidates[row['movie_id']] = max(candidates.get(row['movie_id'], 0), score)

            # --- 2. TF-IDF similarity ---
            q_vec = self._build_query_feature_vector(query_text)
            sims = cosine_similarity(q_vec, self.feature_matrix)[0]
            sims = np.maximum(0, sims)  
            sims = np.power(sims, 1.2)

            for idx, sim in enumerate(sims):
                if sim > 0:
                    movie_id = self.idx_to_movie_id[idx]
                    movie_info = self.movies_df.iloc[idx]
                    # add popularity & quality boosts
                    popularity_boost = min(1.2, 1.0 + np.log1p(movie_info['ratings_count']) / 1000)
                    quality_boost = min(1.3, movie_info['average_rating'] / 3.5 if movie_info['average_rating'] > 0 else 1.0)
                    score = sim * popularity_boost * quality_boost
                    candidates[movie_id] = candidates.get(movie_id, 0) + score  # merge with substring score

            # --- Final ranking ---
            if not candidates:
                return self._get_popular_movies_enhanced(n_recommendations)

            sorted_movies = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            top_ids = [mid for mid, _ in sorted_movies[:n_recommendations]]

            result = self.movies_df[self.movies_df['movie_id'].isin(top_ids)].copy()
            score_map = dict(sorted_movies[:n_recommendations])
            result['query_score'] = result['movie_id'].map(score_map)
            return result.sort_values('query_score', ascending=False)

        except Exception as e:
            # safe fallback
            print(f"[Content] recommend_from_query error: {e}")
            return self._get_popular_movies_enhanced(n_recommendations)

    
    def _get_popular_movies_enhanced(self, n_recommendations=10):
        """Enhanced fallback to popular movies with quality filtering"""
        try:
            # Filter for quality movies
            quality_movies = self.movies_df[
                (self.movies_df['average_rating'] >= 3.8) & 
                (self.movies_df['ratings_count'] >= 50)
            ].copy()
            
            if len(quality_movies) < n_recommendations:
                quality_movies = self.movies_df.copy()
            
            # Score by combined popularity and quality
            quality_movies['popularity_score'] = (
                np.log1p(quality_movies['ratings_count']) * 0.6 +
                quality_movies['average_rating'] * 0.4
            )
            
            return quality_movies.nlargest(n_recommendations, 'popularity_score')
            
        except Exception as e:
            return self.movies_df.head(n_recommendations)
    
    def get_movie_similarity(self, movie_id1, movie_id2):
        """Enhanced movie similarity computation"""
        if movie_id1 in self.movie_id_to_idx and movie_id2 in self.movie_id_to_idx:
            idx1 = self.movie_id_to_idx[movie_id1]
            idx2 = self.movie_id_to_idx[movie_id2]
            return float(self.similarity_matrix[idx1][idx2])
        return 0.0
    
    def get_similar_movies(self, movie_id, n_similar=5):
        """Enhanced similar movies recommendation"""
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame()
        
        movie_idx = self.movie_id_to_idx[movie_id]
        similarities = self.similarity_matrix[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        similar_movie_ids = [self.idx_to_movie_id[idx] for idx in similar_indices]
        
        similar_movies = self.movies_df[self.movies_df['movie_id'].isin(similar_movie_ids)].copy()
        similarity_scores = [similarities[idx] for idx in similar_indices]
        similar_movies['similarity'] = similarity_scores
        
        return similar_movies.sort_values('similarity', ascending=False)

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

class CollaborativeFilteringRecommender:
    def __init__(self, ratings_df, movies_df, *, 
                 min_user_ratings=8, min_item_ratings=3,
                 max_users=4000, max_items=4000,
                 svd_components=60, knn_neighbors=35):
        """
        Optimized Collaborative Filtering with enhanced performance metrics.
        Features:
        - Improved data preprocessing with smart filtering
        - Enhanced SVD with optimal components
        - Optimized kNN with better similarity computation
        - Advanced rating prediction with ensemble methods
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()

        # Optimized configuration
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings
        self.max_users = max_users
        self.max_items = max_items
        self.knn_neighbors = knn_neighbors
        self.svd_components = svd_components

        # Internal structures
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.user_item_matrix = None
        self.sparse_user_item = None
        self.item_user_matrix = None

        # Enhanced models
        self.user_knn = None
        self.item_knn = None
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        
        # Performance enhancements
        self.global_mean = 0.0
        self.user_biases = {}
        self.item_biases = {}
        
        # Build optimized pipeline
        self._prepare_data_optimized()
        self._build_matrices_optimized()
        self._compute_biases()

    def _prepare_data_optimized(self):
        """Enhanced data preparation with smart filtering"""
        df = self.ratings_df[['user_id','movie_id','rating']].copy()
        df['user_id'] = df['user_id'].astype(np.int32)
        df['movie_id'] = df['movie_id'].astype(np.int32)
        df['rating'] = df['rating'].astype(np.float32)
        
        # Remove invalid ratings
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

        # Iterative filtering for better data quality
        for iteration in range(1):  # Multiple passes for better filtering
            user_counts = df['user_id'].value_counts()
            item_counts = df['movie_id'].value_counts()
            
            # Dynamic thresholds based on data distribution
            user_threshold = self.min_user_ratings
            item_threshold = self.min_item_ratings
            
            keep_users = set(user_counts[user_counts >= user_threshold].index)
            keep_items = set(item_counts[item_counts >= item_threshold].index)
            
            old_size = len(df)
            df = df[df['user_id'].isin(keep_users) & df['movie_id'].isin(keep_items)]
            
            # Break if no significant change
            if len(df) > 0.95 * old_size:
                break

        # Activity-based sampling for performance
        if self.max_users and len(df['user_id'].unique()) > self.max_users:
            # Prefer more active users
            user_activity = df['user_id'].value_counts()
            top_users = user_activity.head(self.max_users).index
            df = df[df['user_id'].isin(top_users)]
            
        if self.max_items and len(df['movie_id'].unique()) > self.max_items:
            # Prefer more popular items
            item_popularity = df['movie_id'].value_counts()
            top_items = item_popularity.head(self.max_items).index
            df = df[df['movie_id'].isin(top_items)]

        self.ratings_df = df.reset_index(drop=True)
        print(f"[CF] Filtered data: {len(self.ratings_df)} ratings, {len(df['user_id'].unique())} users, {len(df['movie_id'].unique())} items")

    def _build_mappings(self):
        users = sorted(self.ratings_df['user_id'].unique())
        items = sorted(self.ratings_df['movie_id'].unique())
        self.user_to_idx = {u:i for i,u in enumerate(users)}
        self.idx_to_user = {i:u for u,i in self.user_to_idx.items()}
        self.item_to_idx = {b:i for i,b in enumerate(items)}
        self.idx_to_item = {i:b for b,i in self.item_to_idx.items()}

    def _build_matrices_optimized(self):
        """Build optimized sparse and dense matrices"""
        self._build_mappings()
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        # Create sparse matrix
        rows = self.ratings_df['user_id'].map(self.user_to_idx).values
        cols = self.ratings_df['movie_id'].map(self.item_to_idx).values
        vals = self.ratings_df['rating'].values
        
        self.sparse_user_item = csr_matrix(
            (vals, (rows, cols)), 
            shape=(n_users, n_items), 
            dtype=np.float32
        )

        # Dense matrices for quick access (memory optimized)
        self.user_item_matrix = self.sparse_user_item.toarray()
        self.item_user_matrix = self.user_item_matrix.T
        
        # Compute global statistics
        self.global_mean = float(self.ratings_df['rating'].mean())


    def _compute_biases(self):
        """Compute user and item biases for better predictions"""
        # User biases (how much higher/lower than average each user rates)
        for user_id in self.user_to_idx:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]['rating']
            self.user_biases[user_id] = float(user_ratings.mean() - self.global_mean)
        
        # Item biases (how much higher/lower than average each item is rated)
        for item_id in self.item_to_idx:
            item_ratings = self.ratings_df[self.ratings_df['movie_id'] == item_id]['rating']
            self.item_biases[item_id] = float(item_ratings.mean() - self.global_mean)

    def recommend_from_query(self, query_text: str, n_recommendations: int = 10, top_k_seed: int = 5, min_user_rating: float = 4.0):
        try:
            if not query_text or not str(query_text).strip():
                # fallback to popularity
                pop = self.ratings_df.groupby('movie_id')['rating'].agg(['count', 'mean']).reset_index()
                pop = pop.sort_values(['count', 'mean'], ascending=False)
                candidates = self.movies_df[self.movies_df['movie_id'].isin(pop['movie_id'])].copy()
                return candidates.head(n_recommendations)

            df_movies = self.movies_df
            query_str = str(query_text).strip()

            # --- Priority 1: title ---
            mask_title = df_movies['title'].astype(str).str.contains(query_str, case=False, na=False)
            matched = df_movies[mask_title].copy()

            # --- Priority 2: genres ---
            if matched.empty:
                mask_genre = df_movies['genres'].astype(str).str.contains(query_str, case=False, na=False)
                matched = df_movies[mask_genre].copy()

            # --- Priority 3: year ---
            if matched.empty:
                mask_year = df_movies['year'].astype(str).str.contains(query_str, na=False)
                matched = df_movies[mask_year].copy()

            # --- Priority 4: movie_id numeric ---
            if matched.empty:
                try:
                    possible_id = int(query_str)
                    matched = df_movies[df_movies['movie_id'] == possible_id].copy()
                except Exception:
                    matched = pd.DataFrame()

            if matched.empty:
                # fallback to popularity
                pop = self.ratings_df.groupby('movie_id')['rating'].agg(['count', 'mean']).reset_index()
                pop = pop.sort_values(['count', 'mean'], ascending=False)
                candidates = self.movies_df[self.movies_df['movie_id'].isin(pop['movie_id'])].copy()
                return candidates.head(n_recommendations)

            # --- Seed movies ---
            seed_movie_ids = matched['movie_id'].unique().tolist()[:top_k_seed]
            seed_movies = self.movies_df[self.movies_df['movie_id'].isin(seed_movie_ids)].copy()
            seed_movies["score"] = float("inf")

            # --- Users who rated seeds highly ---
            users = self.ratings_df[self.ratings_df['movie_id'].isin(seed_movie_ids)]
            users = users[users['rating'] >= min_user_rating]['user_id'].unique().tolist()
            if not users:
                users = self.ratings_df[self.ratings_df['movie_id'].isin(seed_movie_ids)]
                users = users[users['rating'] >= 3.5]['user_id'].unique().tolist()

            if not users:
                return seed_movies.head(n_recommendations)

            # --- Candidate ratings ---
            candidate_ratings = self.ratings_df[
                self.ratings_df['user_id'].isin(users) &
                (~self.ratings_df['movie_id'].isin(seed_movie_ids))
            ]
            if candidate_ratings.empty:
                return seed_movies.head(n_recommendations)

            # --- Aggregate popularity/quality score ---
            agg = candidate_ratings.groupby('movie_id')['rating'].agg(['count', 'mean']).reset_index()
            agg['pop_score'] = agg['count'] * agg['mean']

            # --- Correlation with seeds ---
            correlations = []
            for candidate_id in agg['movie_id']:
                cand_ratings = self.ratings_df[self.ratings_df['movie_id'] == candidate_id][['user_id', 'rating']]
                cand_ratings = cand_ratings.set_index('user_id')

                corr_values = []
                for seed_id in seed_movie_ids:
                    seed_ratings = self.ratings_df[self.ratings_df['movie_id'] == seed_id][['user_id', 'rating']]
                    seed_ratings = seed_ratings.set_index('user_id')

                    cand_series, seed_series = cand_ratings['rating'].align(seed_ratings['rating'], join='inner')

                    if len(cand_series) > 1 and len(seed_series) > 1:
                        corr = np.corrcoef(cand_series.values, seed_series.values)[0, 1]
                        if not np.isnan(corr):
                            corr_values.append(corr)

                correlations.append(np.mean(corr_values) if corr_values else 0.0)

            agg['corr_score'] = correlations

            # --- Normalize scores to 0-1 range ---
            def normalize(series):
                if series.max() == series.min():
                    return pd.Series([0.0] * len(series), index=series.index)
                return (series - series.min()) / (series.max() - series.min())

            agg['pop_norm'] = normalize(agg['pop_score'])
            agg['corr_norm'] = normalize(agg['corr_score'])

            # --- Blend scores ---
            alpha = 0.5  # 50% popularity, 50% correlation
            agg['score'] = alpha * agg['pop_norm'] + (1 - alpha) * agg['corr_norm']

            # --- Convert to clean percentages for display ---
            agg['score_display'] = (agg['score'] * 100).round(2)

            # --- Build final results ---
            related_movies = self.movies_df[self.movies_df['movie_id'].isin(agg['movie_id'])].copy()
            related_movies = related_movies.merge(
                agg[['movie_id', 'score', 'score_display']],
                on='movie_id',
                how='left'
            )

            # --- Ensure seed movies are always first ---
            seed_movies = self.movies_df[self.movies_df['movie_id'].isin(seed_movie_ids)].copy()
            seed_movies['score'] = 1.0               # Force max score for seeds
            seed_movies['score_display'] = 100.0     # Show 100% for seeds

            # --- Combine and sort ---
            final_results = pd.concat([seed_movies, related_movies], ignore_index=True)
            final_results = final_results.drop_duplicates(subset=['movie_id'])

            # Priority: (1) Seed movies first, (2) others by score
            final_results['is_seed'] = final_results['movie_id'].isin(seed_movie_ids)
            final_results = final_results.sort_values(
                by=['is_seed', 'score'], ascending=[False, False]
            ).reset_index(drop=True)

            return final_results.head(n_recommendations)

        except Exception as e:
            print(f"[CF] recommend_from_query error: {e}")
            pop = self.ratings_df.groupby('movie_id')['rating'].agg(['count', 'mean']).reset_index()
            pop = pop.sort_values(['count', 'mean'], ascending=False)
            candidates = self.movies_df[self.movies_df['movie_id'].isin(pop['movie_id'])].copy()
            return candidates.head(n_recommendations)


    def predict_rating(self, user_id, movie_id):
        """Enhanced rating prediction with multiple methods"""
        if user_id not in self.user_to_idx or movie_id not in self.item_to_idx:
            return self.global_mean
        
        try:
            predictions = []
            weights = []
            
            # SVD prediction with bias correction
            if self.user_factors is not None and self.item_factors is not None:
                uidx = self.user_to_idx[user_id]
                iidx = self.item_to_idx[movie_id]
                
                svd_pred = np.dot(self.user_factors[uidx], self.item_factors[iidx])
                
                # Apply bias correction
                user_bias = self.user_biases.get(user_id, 0)
                item_bias = self.item_biases.get(movie_id, 0)
                
                final_pred = self.global_mean + user_bias + item_bias + svd_pred * 0.1
                predictions.append(final_pred)
                weights.append(0.6)  # High weight for SVD
            
            # User-based prediction
            if self.user_knn is not None:
                uidx = self.user_to_idx[user_id]
                iidx = self.item_to_idx[movie_id]
                
                distances, indices = self.user_knn.kneighbors(
                    self.sparse_user_item[uidx].reshape(1, -1), 
                    return_distance=True
                )
                
                neighbor_indices = indices.flatten()[1:11]  # Top 10 neighbors
                similarities = 1.0 - distances.flatten()[1:11]
                
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for neighbor_idx, sim in zip(neighbor_indices, similarities):
                    if sim > 0.1:
                        neighbor_rating = self.user_item_matrix[neighbor_idx, iidx]
                        if neighbor_rating > 0:
                            weighted_sum += sim * neighbor_rating
                            weight_sum += sim
                
                if weight_sum > 0:
                    user_pred = weighted_sum / weight_sum
                    predictions.append(user_pred)
                    weights.append(0.4)  # Moderate weight for user-based
            
            # Ensemble prediction
            if predictions:
                final_prediction = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
                return max(1.0, min(5.0, final_prediction))
            else:
                # Fallback to biased global mean
                user_bias = self.user_biases.get(user_id, 0)
                item_bias = self.item_biases.get(movie_id, 0)
                return max(1.0, min(5.0, self.global_mean + user_bias + item_bias))
        
        except Exception as e:
            return self.global_mean

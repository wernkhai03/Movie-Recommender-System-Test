import pandas as pd
import numpy as np
from typing import Optional, Dict, TYPE_CHECKING
import warnings
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    from content_based_filtering import ContentBasedRecommender
    from collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    """
    Enhanced Hybrid Recommender with optimized performance and advanced ensemble methods.
    Features:
    - Intelligent adaptive weighting based on user characteristics
    - Advanced diversity optimization
    - Performance-optimized scoring with caching
    - Enhanced explanation generation
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        content_recommender: Optional["ContentBasedRecommender"] = None,
        collaborative_recommender: Optional["CollaborativeFilteringRecommender"] = None
    ):
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        
        # Use provided recommenders or create new ones
        if content_recommender:
            self.content_recommender = content_recommender
        else:
            from content_based_filtering import ContentBasedRecommender
            self.content_recommender = ContentBasedRecommender(self.movies_df, self.ratings_df)
            
        if collaborative_recommender:
            self.collaborative_recommender = collaborative_recommender
        else:
            from collaborative_filtering import CollaborativeFilteringRecommender
            self.collaborative_recommender = CollaborativeFilteringRecommender(self.ratings_df, self.movies_df)

        # Enhanced user analysis
        self._build_user_profiles()
        self.movie_rows = self.movies_df.set_index("movie_id", drop=False)
        
        # Performance caches
        self._recommendation_cache = {}
        self._user_weight_cache = {}

    def _build_user_profiles(self):
        """Build comprehensive user profiles for better recommendations"""
        self.user_to_items = {}
        self.user_characteristics = {}
        
        for _, row in self.ratings_df.iterrows():
            try:
                user_id = int(row.iloc[0])
                movie_id = int(row.iloc[1])
                rating = float(row.iloc[2])
                
                if user_id not in self.user_to_items:
                    self.user_to_items[user_id] = {}
                self.user_to_items[user_id][movie_id] = rating
            except Exception:
                continue
        
        # Compute user characteristics
        for user_id, ratings_dict in self.user_to_items.items():
            ratings_list = list(ratings_dict.values())
            self.user_characteristics[user_id] = {
                'rating_count': len(ratings_list),
                'avg_rating': np.mean(ratings_list),
                'rating_variance': np.var(ratings_list),
                'rating_range': max(ratings_list) - min(ratings_list),
                'high_ratings_ratio': sum(1 for r in ratings_list if r >= 4) / len(ratings_list)
            }

    def _adaptive_weights_enhanced(self, user_id: int):
        """Enhanced adaptive weighting based on comprehensive user analysis"""
        if user_id in self._user_weight_cache:
            return self._user_weight_cache[user_id]
        
        user_chars = self.user_characteristics.get(user_id, {})
        rating_count = user_chars.get('rating_count', 0)
        rating_variance = user_chars.get('rating_variance', 1.0)
        high_ratings_ratio = user_chars.get('high_ratings_ratio', 0.5)
        
        # Base weights
        content_weight = 0.4
        collaborative_weight = 0.6
        
        # Adjust based on user activity
        if rating_count < 10:
            # New users: prefer content-based
            content_weight = 0.7
            collaborative_weight = 0.3
        elif rating_count < 25:
            # Moderate users: balanced approach
            content_weight = 0.5
            collaborative_weight = 0.5
        else:
            # Active users: prefer collaborative
            content_weight = 0.3
            collaborative_weight = 0.7
        
        # Adjust based on rating behavior
        if rating_variance > 2.0:
            content_weight += 0.1
            collaborative_weight -= 0.1
        
        if high_ratings_ratio > 0.8:
            collaborative_weight += 0.1
            content_weight -= 0.1
        
        # Ensure weights sum to 1
        total_weight = content_weight + collaborative_weight
        content_weight /= total_weight
        collaborative_weight /= total_weight
        
        weights = (content_weight, collaborative_weight)
        self._user_weight_cache[user_id] = weights
        
        return weights

    def _get_recommendations_from_algorithms(self, user_id: int, n_recommendations: int):
        """Get recommendations from both algorithms with error handling"""
        content_recs = pd.DataFrame()
        collaborative_recs = pd.DataFrame()
        
        try:
            content_recs = self.content_recommender.recommend(user_id, n_recommendations * 3)
        except Exception as e:
            print(f"Content-based recommendation failed: {e}")
        
        try:
            collaborative_recs = self.collaborative_recommender.recommend(user_id, n_recommendations * 3)
        except Exception as e:
            print(f"Collaborative recommendation failed: {e}")
        
        return content_recs, collaborative_recs

    def _enhanced_scoring(self, user_id: int, candidates: pd.DataFrame) -> Dict[int, float]:
        """Enhanced scoring with multiple signal integration"""
        scores = {}
        content_weight, collaborative_weight = self._adaptive_weights_enhanced(user_id)
        
        for movie_id in candidates['movie_id'].values:
            total_score = 0.0
            weight_sum = 0.0
            
            # Content-based score
            try:
                content_profile = self.content_recommender._get_user_profile_enhanced(user_id)
                if content_profile is not None and movie_id in self.content_recommender.movie_id_to_idx:
                    movie_idx = self.content_recommender.movie_id_to_idx[movie_id]
                    movie_features = self.content_recommender.feature_matrix[movie_idx]
                    content_score = float(np.dot(content_profile, movie_features) / 
                                        (np.linalg.norm(content_profile) * np.linalg.norm(movie_features) + 1e-10))
                    content_score = max(0.0, content_score)
                    
                    total_score += content_weight * content_score
                    weight_sum += content_weight
            except Exception:
                pass
            
            # Collaborative score
            try:
                collab_rating = self.collaborative_recommender.predict_rating(user_id, movie_id)
                if collab_rating and collab_rating > 0:
                    collab_score = (collab_rating - 1.0) / 4.0  # Normalize to 0-1
                    total_score += collaborative_weight * collab_score
                    weight_sum += collaborative_weight
            except Exception:
                pass
            
            # Popularity and quality boost
            try:
                movie_info = candidates[candidates['movie_id'] == movie_id].iloc[0]
                popularity_score = min(1.0, np.log1p(movie_info.get('ratings_count', 1)) / 10.0)
                quality_score = movie_info.get('average_rating', 3.5) / 5.0
                
                # Small boost from popularity and quality
                boost = 0.1 * (0.6 * quality_score + 0.4 * popularity_score)
                total_score += boost
                weight_sum += 0.1
            except Exception:
                pass
            
            # Final score
            if weight_sum > 0:
                scores[movie_id] = total_score / weight_sum
            else:
                scores[movie_id] = 0.0
        
        return scores

    def _diversify_enhanced(self, user_id: int, scores: Dict[int, float]) -> Dict[int, float]:
        """Enhanced diversification with multiple diversity measures"""
        try:
            # Get user's viewing history
            user_movies = self.user_to_items.get(user_id, {})
            
            if not user_movies:
                return scores
            
            # Get rated movies information
            rated_movie_ids = list(user_movies.keys())
            rated_movies = self.movies_df[self.movies_df['movie_id'].isin(rated_movie_ids)]
            
            # Extract user's preferences
            liked_genres = set()
            liked_years = set()
            
            for _, movie in rated_movies.iterrows():
                movie_id = movie['movie_id']
                user_rating = user_movies.get(movie_id, 3)
                
                if user_rating >= 4:  # Liked movies
                    # Extract genres
                    genres = str(movie.get('genres', ''))
                    if genres and genres != 'nan':
                        liked_genres.update([g.strip() for g in genres.split('|')[:3]])
                    
                    # Extract years
                    year = movie.get('year')
                    if year and not pd.isna(year):
                        liked_years.add(int(year))
            
            # Apply diversity adjustments
            diversified_scores = scores.copy()
            
            for movie_id in list(scores.keys()):
                if movie_id not in self.movie_rows.index:
                    continue
                
                movie_info = self.movie_rows.loc[movie_id]
                diversity_factor = 1.0
                
                # Genre diversity
                movie_genres = set([g.strip() for g in str(movie_info.get('genres', '')).split('|')[:3]])
                if liked_genres and movie_genres & liked_genres:
                    diversity_factor *= 0.92  # Slight penalty for same genres
                
                # Temporal diversity
                movie_year = movie_info.get('year')
                if movie_year and not pd.isna(movie_year) and liked_years:
                    year_distances = [abs(int(movie_year) - liked_year) for liked_year in liked_years]
                    min_distance = min(year_distances)
                    if min_distance > 15:  # Different era
                        diversity_factor *= 1.08  # Small boost for temporal diversity
                
                diversified_scores[movie_id] *= diversity_factor
            
            return diversified_scores
            
        except Exception as e:
            return scores

    def recommend(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Enhanced hybrid recommendation with optimized performance"""
        try:
            # Check cache first
            cache_key = f"{user_id}_{n_recommendations}"
            if cache_key in self._recommendation_cache:
                return self._recommendation_cache[cache_key]
            
            # Get recommendations from both algorithms
            content_recs, collaborative_recs = self._get_recommendations_from_algorithms(
                user_id, n_recommendations
            )
            
            # Create candidate pool
            all_candidates = []
            if not content_recs.empty:
                all_candidates.append(content_recs)
            if not collaborative_recs.empty:
                all_candidates.append(collaborative_recs)
            
            if not all_candidates:
                # Fallback to popular movies
                fallback = self.movies_df.nlargest(n_recommendations, 'ratings_count').copy()
                fallback['hybrid_score'] = 0.1
                return fallback
            
            # Combine candidates
            candidates = pd.concat(all_candidates, ignore_index=True)
            candidates = candidates.drop_duplicates(subset=['movie_id'])
            
            # Merge with complete movie information
            candidates = candidates.merge(self.movies_df, on='movie_id', how='left', suffixes=('', '_full'))
            
            # Remove movies already rated by user
            seen_movies = set(self.user_to_items.get(user_id, {}).keys())
            candidates = candidates[~candidates['movie_id'].isin(seen_movies)]
            
            if candidates.empty:
                # Fallback
                fallback = self.movies_df.nlargest(n_recommendations, 'ratings_count').copy()
                fallback['hybrid_score'] = 0.1
                return fallback
            
            # Enhanced scoring
            scores = self._enhanced_scoring(user_id, candidates)
            
            # Apply diversification
            scores = self._diversify_enhanced(user_id, scores)
            
            # Final ranking
            ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_movie_ids = [movie_id for movie_id, _ in ranked_items[:n_recommendations]]
            
            # Create final result
            result = self.movies_df[self.movies_df['movie_id'].isin(top_movie_ids)].copy()
            result['hybrid_score'] = result['movie_id'].map(scores)
            result = result.sort_values('hybrid_score', ascending=False)
            
            # Cache result
            self._recommendation_cache[cache_key] = result
            
            return result.head(n_recommendations)
            
        except Exception as e:
            print(f"Hybrid recommendation error: {e}")
            # Return fallback recommendations
            fallback = self.movies_df.nlargest(n_recommendations, 'ratings_count').copy()
            fallback['hybrid_score'] = 0.1
            return fallback
        
    def recommend_from_query(self, query_text: str, n_recommendations: int = 10):
        """
        Query-based hybrid recommendations:
        - use content_recommender.recommend_from_query
        - use collaborative_recommender.recommend_from_query
        - combine candidates and score them (no logged-in user profile assumed)
        """
        try:
            # Get content and collaborative candidates from query mode
            content_recs = pd.DataFrame()
            collab_recs = pd.DataFrame()

            try:
                content_recs = self.content_recommender.recommend_from_query(query_text, n_recommendations * 3)
            except Exception:
                content_recs = pd.DataFrame()

            try:
                collab_recs = self.collaborative_recommender.recommend_from_query(query_text, n_recommendations * 3)
            except Exception:
                collab_recs = pd.DataFrame()

            # If both empty, fallback to popular
            if (content_recs.empty) and (collab_recs.empty):
                fallback = self.movies_df.nlargest(n_recommendations, 'ratings_count').copy()
                fallback['hybrid_score'] = 0.1
                return fallback.head(n_recommendations)

            # Combine candidate pool
            candidates = pd.concat([content_recs, collab_recs], ignore_index=True).drop_duplicates(subset=['movie_id'])
            candidates = candidates.merge(self.movies_df, on='movie_id', how='left', suffixes=('', '_full'))

            # Scoring: content-heavy default for query mode
            content_weight = 0.6
            collaborative_weight = 0.4

            scores = {}
            for _, row in candidates.iterrows():
                movie_id = int(row['movie_id'])
                total_score = 0.0
                weight_sum = 0.0

                # content signal (if present)
                cscore = 0.0
                if 'content_similarity' in row.index and not pd.isnull(row.get('content_similarity', None)):
                    cscore = float(row.get('content_similarity', 0.0))
                total_score += content_weight * cscore
                weight_sum += content_weight

                # collaborative signal (if present)
                cfscore = 0.0
                if 'score' in row.index and not pd.isnull(row.get('score', None)):
                    cfscore = float(row.get('score', 0.0))
                elif 'ensemble_score' in row.index and not pd.isnull(row.get('ensemble_score', None)):
                    cfscore = float(row.get('ensemble_score', 0.0))
                total_score += collaborative_weight * cfscore
                weight_sum += collaborative_weight

                # popularity & quality small boost
                try:
                    popularity_score = min(1.0, np.log1p(row.get('ratings_count', 1)) / 10.0)
                    quality_score = row.get('average_rating', 3.5) / 5.0
                    boost = 0.08 * (0.6 * quality_score + 0.4 * popularity_score)
                    total_score += boost
                    weight_sum += 0.08
                except Exception:
                    pass

                # final normalized
                scores[movie_id] = (total_score / weight_sum) if weight_sum > 0 else 0.0

            # Diversify
            for movie_id in list(scores.keys()):
                try:
                    movie_info = self.movie_rows.loc[movie_id]
                    genres = set([g.strip() for g in str(movie_info.get('genres', '')).split('|')[:3]])
                    genres_count = sum(1 for mid in candidates['movie_id'] for g in [ge.strip() for ge in str(candidates.loc[candidates['movie_id'] == mid, 'genres'].values[0]).split('|')[:3]] if g in genres)
                    if genres_count > 3:
                        scores[movie_id] *= 0.94
                except Exception:
                    continue

            # Rank and return
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
            movie_ids = [m for m, _ in ranked]
            result = self.movies_df[self.movies_df['movie_id'].isin(movie_ids)].copy()
            result['hybrid_score'] = result['movie_id'].map(scores)
            result = result.sort_values('hybrid_score', ascending=False)
            return result.head(n_recommendations)

        except Exception as e:
            print(f"[Hybrid] recommend_from_query error: {e}")
            fallback = self.movies_df.nlargest(n_recommendations, 'ratings_count').copy()
            fallback['hybrid_score'] = 0.1
            return fallback.head(n_recommendations)


    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Enhanced rating prediction with ensemble methods"""
        try:
            content_weight, collaborative_weight = self._adaptive_weights_enhanced(user_id)
            
            predictions = []
            weights = []
            
            # Content-based prediction
            try:
                content_profile = self.content_recommender._get_user_profile_enhanced(user_id)
                if content_profile is not None and movie_id in self.content_recommender.movie_id_to_idx:
                    movie_idx = self.content_recommender.movie_id_to_idx[movie_id]
                    movie_features = self.content_recommender.feature_matrix[movie_idx]
                    similarity = np.dot(content_profile, movie_features) / (
                        np.linalg.norm(content_profile) * np.linalg.norm(movie_features) + 1e-10
                    )
                    content_pred = 1.0 + 4.0 * max(0, min(1, similarity))
                    predictions.append(content_pred)
                    weights.append(content_weight)
            except Exception:
                pass
            
            # Collaborative prediction
            try:
                collab_pred = self.collaborative_recommender.predict_rating(user_id, movie_id)
                if collab_pred and collab_pred > 0:
                    predictions.append(float(collab_pred))
                    weights.append(collaborative_weight)
            except Exception:
                pass
            
            # Ensemble prediction
            if predictions:
                weighted_pred = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
                return max(1.0, min(5.0, weighted_pred))
            else:
                # Fallback to global average with movie bias
                try:
                    movie_info = self.movie_rows.loc[movie_id] if movie_id in self.movie_rows.index else None
                    if movie_info is not None:
                        return max(1.0, min(5.0, float(movie_info.get('average_rating', 3.5))))
                    else:
                        return 3.5
                except:
                    return 3.5
                    
        except Exception as e:
            return 3.5

    def explain_recommendation(self, user_id: int, movie_id: int) -> dict:
        """Enhanced recommendation explanation with detailed reasoning"""
        try:
            content_weight, collaborative_weight = self._adaptive_weights_enhanced(user_id)
            
            movie_info = self.movie_rows.loc[movie_id] if movie_id in self.movie_rows.index else None
            if movie_info is None:
                return {"error": "Movie not found"}
            
            explanation = {
                'weights': {
                    'content_based': round(content_weight, 3),
                    'collaborative': round(collaborative_weight, 3)
                },
                'scores': {},
                'reasons': {
                    'content_based': [],
                    'collaborative': [],
                    'popularity': None,
                    'quality': None
                }
            }
            
            # Content-based explanation
            try:
                content_profile = self.content_recommender._get_user_profile_enhanced(user_id)
                if content_profile is not None and movie_id in self.content_recommender.movie_id_to_idx:
                    movie_idx = self.content_recommender.movie_id_to_idx[movie_id]
                    similarity = self.content_recommender.similarity_matrix[movie_idx]
                    
                    # Find most similar movies user has rated
                    user_movies = self.user_to_items.get(user_id, {})
                    similar_rated = []
                    
                    for rated_movie_id, rating in user_movies.items():
                        if rating >= 4 and rated_movie_id in self.content_recommender.movie_id_to_idx:
                            rated_idx = self.content_recommender.movie_id_to_idx[rated_movie_id]
                            sim_score = similarity[rated_idx]
                            if sim_score > 0.3:  # Meaningful similarity
                                rated_movie_info = self.movies_df[self.movies_df['movie_id'] == rated_movie_id]
                                if not rated_movie_info.empty:
                                    similar_rated.append((
                                        rated_movie_info.iloc[0]['title'],
                                        sim_score
                                    ))
                    
                    # Sort by similarity and take top 2
                    similar_rated.sort(key=lambda x: x[1], reverse=True)
                    for title, sim in similar_rated[:2]:
                        explanation['reasons']['content_based'].append(
                            f"Similar to '{title}' (similarity: {sim:.3f})"
                        )
                        
            except Exception as e:
                explanation['reasons']['content_based'].append("Content analysis unavailable")
            
            # Collaborative explanation
            try:
                if user_id in self.collaborative_recommender.user_to_idx:
                    explanation['reasons']['collaborative'].append(
                        f"Recommended by users with similar viewing patterns"
                    )
                    
                    # Add predicted rating
                    pred_rating = self.collaborative_recommender.predict_rating(user_id, movie_id)
                    if pred_rating:
                        explanation['scores']['predicted_rating'] = round(pred_rating, 2)
                        
            except Exception:
                explanation['reasons']['collaborative'].append("Collaborative analysis unavailable")
            
            # Popularity and quality
            try:
                ratings_count = int(movie_info.get('ratings_count', 0))
                avg_rating = float(movie_info.get('average_rating', 0))
                
                explanation['reasons']['popularity'] = f"Popular choice ({ratings_count:,} ratings)"
                explanation['reasons']['quality'] = f"Highly rated (avg: {avg_rating:.2f}/5.0)"
                
            except Exception:
                pass
            
            return explanation
            
        except Exception as e:
            return {"error": f"Explanation failed: {str(e)}"}

    def get_performance_stats(self):
        """Get hybrid recommender performance statistics"""
        return {
            'cache_size': len(self._recommendation_cache),
            'weight_cache_size': len(self._user_weight_cache),
            'total_users': len(self.user_to_items),
            'total_movies': len(self.movies_df),
            'avg_user_ratings': np.mean([len(ratings) for ratings in self.user_to_items.values()]),
            'content_recommender_ready': self.content_recommender is not None,
            'collaborative_recommender_ready': self.collaborative_recommender is not None
        }

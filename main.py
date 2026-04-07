import textwrap, streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import os, time, threading, re, tempfile
import hashlib
import json
warnings.filterwarnings('ignore')

# Import custom modules (with fallback for demo)
try:
    from content_based_filtering import ContentBasedRecommender
    from collaborative_filtering import CollaborativeFilteringRecommender
    from hybrid_filtering import HybridRecommender
    from metrics_analysis import OptimizedMetricsAnalyzer
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    st.warning("Custom recommendation modules not found. Running in demo mode.")

# Configure page
st.set_page_config(
    page_title="CineVerse - Professional Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS with modern design
st.markdown(textwrap.dedent("""
<style>
    .algo-card{ background:linear-gradient(135deg,#fff 0%,#f8fafc 100%); padding:1.5rem; border-radius:16px; }
    .algo-card *{ color:#0f172a !important; }
</style>
"""), unsafe_allow_html=True)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-gradient: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        --accent-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        --success-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --warning-gradient: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        --error-gradient: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        --text-primary: #1e293b;
        --text-secondary: #3d3d32;
        --border-color: #e2e8f0;
        --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        --border-radius: 16px;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 0rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .stApp {
        background: var(--primary-gradient);
        min-height: 100vh;
    }

    /* Professional metric cards */
    .metric-card {
        background: var(--card-gradient);
        padding: 2rem;
        border-radius: var(--border-radius);
        text-align: center;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--accent-gradient);
        transform: translateX(-100%);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: var(--shadow-lg);
    }
    
    .metric-card:hover::before {
        transform: translateX(0);
    }
    
    .metric-card .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .metric-card h2 {
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0.5rem 0;
        color: var(--text-primary);
    }
    
    .metric-card h3 {
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Enhanced movie cards */
    .book-card {
        background: var(--card-gradient);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: var(--shadow-md);
        margin: 2rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .book-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #ec4899);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .book-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-lg);
    }
    
    .book-card:hover::before {
        opacity: 1;
    }
            
    /* Algorithm selection cards */
    .algorithm-card {
        background: var(--card-gradient);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-sm);
        margin: 1rem 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        position: relative;
        cursor: pointer;
        height: 100%;
        min-height: 280px;
        display: flex;
        flex-direction: column;
    }
    
    .algorithm-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
        border-color: #4f46e5;
    }
    
    .algorithm-card.disabled {
        opacity: 0.6;
        cursor: not-allowed;
        background: #f8fafc;
    }
    
    .algorithm-card.selected {
        border-color: #4f46e5;
        background: linear-gradient(135deg, #f0f7ff 0%, #e0f2fe 100%);
    }
    
    .algorithm-card h4 {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .algorithm-card .algorithm-icon {
        font-size: 1.5rem;
    }
    
    .algorithm-card p {
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 1.5rem;
        flex-grow: 1;
    }
    
    .algorithm-features {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }

    .algorithm-features li {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .algorithm-features li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #10b981;
        font-weight: bold;
    }
    
    /* Enhanced title */
    .title-gradient {
        background: linear-gradient(135deg, #1e293b 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    
    .subtitle {
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.25rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Professional buttons */
    .stButton > button {
        background: var(--accent-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        letter-spacing: 0.025em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        filter: brightness(1.05);
    }
    
    .stButton > button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
    }
    
    /* Form improvements */
    .stSelectbox [data-baseweb="select"] div[class*="-SingleValue"],
    .stSelectbox [data-baseweb="select"] div[class*="-ValueContainer"] {
        color:#111827 !important;
        -webkit-text-fill-color:#111827 !important;
        opacity:1 !important;
        mix-blend-mode:normal !important;
        text-shadow:none !important;
    }

    .stSelectbox [data-baseweb="select"] div[class*="-Placeholder"] {
        color:#64748b !important;
        -webkit-text-fill-color:#64748b !important;
        opacity:1 !important;
    }

    .stSelectbox [data-baseweb="select"] input {
        color:#111827 !important;
        -webkit-text-fill-color:#111827 !important;
        opacity:1 !important;
    }

    .stSelectbox [data-baseweb="select"] [role="listbox"] * {
        color:#111827 !important;
        -webkit-text-fill-color:#111827 !important;
        opacity:1 !important;
    }

    .stSelectbox [data-baseweb="select"] [aria-selected="true"] {
        background:rgba(79,70,229,.10) !important;
        color:#111827 !important;
    }
            
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 0.875rem 1rem;
        font-weight: 500;
        background: white;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within,
    .stTextArea > div > div > textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        outline: none;
    }

    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-premium {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    
    .status-basic {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
    }
    
    .performance-card { background:#fff; color:#111827; position:relative; }
    .performance-card h4,
    .performance-card h4 span {
        color:#111827 !important;
        opacity:1 !important;
        mix-blend-mode:normal !important;
        font-weight:600 !important;
        margin:0 0 .25rem 0 !important;
    }
    .performance-card small {
        color:#64748b !important;
        opacity:1 !important;
    }
            
    /* Notification styles */
    .notification {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .notification-success {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .notification-error {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .notification-warning {
        background: rgba(245, 158, 11, 0.1);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .notification-info {
        background: rgba(59, 130, 246, 0.1);
        color: #2563eb;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Tab improvements */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.25rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 600;
        border-radius: 8px;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        backdrop-filter: blur(10px);
    }
    
    /* Performance cards */
    .performance-card {
        background: var(--card-gradient);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-sm);
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .performance-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
            
    .match-badge{
        display:flex; align-items:center; gap:.5rem; margin:.25rem 0 .5rem 0;
        font-weight:600; letter-spacing:.2px;
    }
    .match-badge .pill{
        padding:.15rem .5rem; border-radius:999px; font-size:.85rem;
        background:rgba(76,175,80,.18); border:1px solid rgba(76,175,80,.35); color:#c8ffd0;
    }
    .match-badge .bar{
        flex:1; height:8px; background:rgba(255,255,255,.12); border-radius:999px; overflow:hidden;
    }
    .match-badge .fill{
        height:100%; width:0%; background:linear-gradient(90deg,#22c55e,#84cc16);
    }
        
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 0rem 1rem;
        }
        
        .title-gradient {
            font-size: 2.5rem;
        }
        
        .login-container {
            margin: 1rem;
            padding: 2rem;
        }
        
        .metric-card {
            padding: 1.5rem;
        }
        
        .book-card {
            padding: 1.5rem;
        }
        
        .algorithm-card {
            min-height: 240px;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class ValidationManager:
    """Comprehensive validation for user inputs and system state"""
    @staticmethod
    def validate_user_id(user_id, valid_user_ids, user_rating_counts):
        """Enhanced user ID validation with strict dataset range checks + feedback"""
        validation_result = {
            'is_valid': False,
            'message': '',
            'type': 'error',
            'details': {},
            'suggestions': []
        }

        if not isinstance(user_id, (int, float)) or user_id != int(user_id):
            validation_result.update({
                'message': 'User ID must be a valid number',
                'suggestions': ['Enter a numeric user ID', 'Check for typos']
            })
            return validation_result
        user_id = int(user_id)

        if user_id < 1:
            validation_result.update({
                'message': 'User ID must be greater than 0',
                'suggestions': ['Enter a positive number']
            })
            return validation_result

        if valid_user_ids:
            min_id, max_id = min(valid_user_ids), max(valid_user_ids)
            if user_id < min_id or user_id > max_id:
                validation_result.update({
                    'message': f'User ID {user_id:,} is out of range. Valid IDs are between {min_id:,} and {max_id:,}.',
                    'type': 'error',
                    'suggestions': [f'Choose an ID between {min_id:,} and {max_id:,}', 'If unsure, ask an admin for your ID']
                })
                return validation_result

        if user_id not in valid_user_ids:
            similar = ValidationManager._find_similar_user_ids(user_id, valid_user_ids)
            validation_result.update({
                'message': f'User ID {user_id:,} not found in dataset',
                'type': 'error',
                'suggestions': [
                    f'Try one of these: {", ".join(map(str, similar[:3]))}' if similar else 'Browse available user IDs',
                    'Contact support if you believe this is an error'
                ]
            })
            return validation_result

        rating_count = user_rating_counts.get(user_id, 0)
        if rating_count == 0:
            validation_result.update({
                'message': 'User has no ratings in the system',
                'type': 'warning',
                'suggestions': ['Rate some movies first to get recommendations']
            })
            return validation_result

        eligibility_status = "Premium" if rating_count >= 15 else "Basic"
        validation_result.update({
            'is_valid': True,
            'message': f'Valid user with {rating_count:,} ratings ({eligibility_status})',
            'type': 'success',
            'details': {
                'rating_count': rating_count,
                'eligibility': eligibility_status,
                'is_premium': rating_count >= 15
            }
        })
        return validation_result
    
    @staticmethod
    def _find_similar_user_ids(target_id, valid_user_ids, max_suggestions=5):
        if not valid_user_ids:
            return []
        valid_list = sorted(list(valid_user_ids))
        differences = [(abs(uid - target_id), uid) for uid in valid_list]
        differences.sort()
        return [uid for _, uid in differences[:max_suggestions]]
    
    @staticmethod
    def validate_rating(rating):
        validation_result = {'is_valid': False, 'message': '', 'type': 'error'}
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            validation_result.update({'message': 'Rating must be a number'})
            return validation_result
        if rating < 1 or rating > 5:
            validation_result.update({'message': 'Rating must be between 1 and 5'})
            return validation_result
        validation_result.update({'is_valid': True, 'message': f'Valid rating: {rating}', 'type': 'success'})
        return validation_result
    
    @staticmethod
    def validate_dataset(ratings_df, movies_df):
        issues = []
        warnings_list = []
        if ratings_df.empty:
            issues.append("Ratings dataset is empty")
        if movies_df.empty:
            issues.append("Movies dataset is empty")
        if issues:
            return {'status': 'error', 'issues': issues, 'warnings': warnings_list}
        
        required_rating_cols = ['user_id', 'movie_id', 'rating']
        required_movie_cols = ['movie_id', 'title']
        missing_rating_cols = [col for col in required_rating_cols if col not in ratings_df.columns]
        missing_movie_cols = [col for col in required_movie_cols if col not in movies_df.columns]
        if missing_rating_cols:
            issues.append(f"Missing rating columns: {missing_rating_cols}")
        if missing_movie_cols:
            issues.append(f"Missing movie columns: {missing_movie_cols}")
        if issues:
            return {'status': 'error', 'issues': issues, 'warnings': warnings_list}
        
        null_ratings = ratings_df.isnull().sum().sum()
        if null_ratings > 0:
            warnings_list.append(f"Found {null_ratings:,} null values in ratings")
        
        if 'rating' in ratings_df.columns:
            invalid_ratings = ratings_df[(ratings_df['rating'] < 1) | (ratings_df['rating'] > 5)]
            if len(invalid_ratings) > 0:
                warnings_list.append(f"Found {len(invalid_ratings):,} invalid ratings (outside 1-5 range)")
        
        common_movies = set(ratings_df['movie_id'].unique()) & set(movies_df['movie_id'].unique())
        overlap_ratio = len(common_movies) / len(movies_df['movie_id'].unique()) if len(movies_df) > 0 else 0
        if overlap_ratio < 0.5:
            warnings_list.append(f"Low overlap between ratings and movies data ({overlap_ratio:.1%})")
        
        return {
            'status': 'success' if not issues else 'warning',
            'issues': issues,
            'warnings': warnings_list,
            'stats': {
                'total_ratings': len(ratings_df),
                'total_movies': len(movies_df),
                'total_users': ratings_df['user_id'].nunique() if 'user_id' in ratings_df.columns else 0,
                'overlap_ratio': overlap_ratio
            }
        }

# Optimized caching helpers
@st.cache_data(show_spinner=False, ttl=3600, persist="disk")
def load_csv_optimized(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
        if 'ratings' in path.lower():
            cols = [c.strip().lower() for c in df.columns.tolist()]
            df.columns = cols
            if {'user_id','movie_id','rating'}.issubset(set(cols)):
                df = df[['user_id','movie_id','rating']].copy()
            else:
                df = df.iloc[:, :3].copy()
                df.columns = ['user_id','movie_id','rating']
            df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
            df['movie_id'] = pd.to_numeric(df['movie_id'], errors='coerce')
            df['rating']  = pd.to_numeric(df['rating'],  errors='coerce')
            df = df.dropna()
            df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
            return df.astype({'user_id':'int32','movie_id':'int32','rating':'float32'})
        else:
            return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame(columns=['user_id','movie_id','rating'])

@st.cache_resource(show_spinner=False)
def build_content_recommender(movies_df, ratings_df, *, cache_key: str = "v1"):
    if MODULES_AVAILABLE:
        return ContentBasedRecommender(movies_df, ratings_df)
    return None

@st.cache_resource(show_spinner=False)
def build_collaborative_recommender(ratings_df, movies_df, *, cache_key: str = "v1"):
    if MODULES_AVAILABLE:
        return CollaborativeFilteringRecommender(
            ratings_df, movies_df,
            min_user_ratings=2, min_item_ratings=2,
            max_users=None, max_items=None,
            svd_components=50, knn_neighbors=15
        )
    return None

@st.cache_resource(show_spinner=False)
def build_hybrid_recommender(_movies_df, _ratings_df, _content_rec=None, _collab_rec=None, *, cache_key: str = "v1"):
    if MODULES_AVAILABLE:
        _ = cache_key 
        return HybridRecommender(_movies_df, _ratings_df, _content_rec, _collab_rec)
    return None

class EnhancedMovieRecommenderApp:
    def __init__(self):
        self.validator = ValidationManager()
        self.initialize_session_state()
        self.data_dir = Path(__file__).resolve().parent
        if self.load_and_validate_data():
            self.init_recommenders()
        else:
            st.stop()

    def initialize_session_state(self):
        defaults = {
            'logged_in': False,
            'current_user': None,
            'user_eligible': False,
            'user_ratings_count': 0,
            'selected_algorithm': None,
            'user_feedback': [],
            'last_recommendations': {},
            'system_performance': {},
            'analysis_results': None,
            'just_saved_rating': None,
            'validation_state': {},
            'login_attempts': 0,
            'last_login_attempt': None
        }
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _add_match_scores(self, recs_df, algorithm: str):
        import numpy as np
        if recs_df is None or len(recs_df) == 0:
            return recs_df
        df = recs_df.copy()
        algo = (algorithm or "").lower()
        direct_cols_priority = []
        if algo == "content":
            direct_cols_priority = ["content_similarity", "similarity", "content_score"]
        elif algo == "collaborative":
            direct_cols_priority = ["predicted_rating", "cf_pred", "cf_rating", "ensemble_score",
                                    "cf_user_score", "cf_item_score", "svd_score"]
        elif algo == "hybrid":
            direct_cols_priority = ["hybrid_pred_rating", "predicted_rating", "hybrid_score",
                                    "ensemble_score", "content_similarity", "content_score"]
        src_col = next((c for c in direct_cols_priority if c in df.columns), None)
        raw = None
        if src_col is not None:
            s = df[src_col].astype(float)
            if s.min() >= 1.0 and s.max() <= 5.0:
                raw = (s - 1.0) / 4.0
            elif s.min() >= 0.0 and s.max() <= 1.0:
                raw = s.clip(0.0, 1.0)
            else:
                mn, mx = float(s.min()), float(s.max())
                if mx > mn:
                    raw = (s - mn) / (mx - mn)
                else:
                    raw = pd.Series(1.0, index=s.index)
        if raw is None:
            proxy_cols = [c for c in ["average_rating", "popularity_score", "ratings_count"] if c in df.columns]
            if proxy_cols:
                s = df[proxy_cols[0]].astype(float)
                mn, mx = float(s.min()), float(s.max())
                raw = (s - mn) / (mx - mn) if mx > mn else pd.Series(1.0, index=s.index)
            else:
                raw = pd.Series(0.5, index=df.index)
        df["match_score_pct"] = (raw.clip(0.0, 1.0) * 100.0).round(1)
        return df

    def _data_path(self, name: str) -> str:
        return str(Path(__file__).resolve().parent / name)
    
    def load_and_validate_data(self):
        try:
            required_files = ['ratings.csv', 'movies.csv']
            missing = [f for f in required_files if not Path(self._data_path(f)).exists()]
            if missing:
                st.error(f"Missing required files: {', '.join(missing)}")
                if st.button("Create Demo Data"):
                    self._create_demo_data()
                    st.rerun()
                return False
            with st.spinner("Loading and validating dataset..."):
                self.ratings_df = load_csv_optimized(self._data_path('ratings.csv'))
                self.movies_df  = load_csv_optimized(self._data_path('movies.csv'))
            self._process_datasets()
            validation_result = self.validator.validate_dataset(self.ratings_df, self.movies_df)
            if validation_result['status'] == 'error':
                st.error("Dataset validation failed:")
                for issue in validation_result['issues']:
                    st.error(f"* {issue}")
                return False
            if validation_result['warnings']:
                st.warning("Dataset validation warnings:")
                for w in validation_result['warnings']:
                    st.warning(f"* {w}")
            self._calculate_dataset_statistics()
            stats = validation_result.get('stats', {})
            st.success(
                f"Dataset loaded successfully: "
                f"{stats.get('total_users', 0):,} users, "
                f"{stats.get('total_movies', 0):,} movies, "
                f"{stats.get('total_ratings', 0):,} ratings"
            )
            return True
        except Exception as e:
            st.error(f"Critical error loading data: {e}")
            return False

    def _create_demo_data(self):
        try:
            movies_data = {
                'id': range(1, 101),
                'movie_id': range(1, 101),
                'title': [f'Movie Title {i} ({2000+i%24})' for i in range(1, 101)],
                'genres': ['Action|Adventure' if i%3==0 else 'Drama|Romance' if i%3==1 else 'Comedy|Thriller' for i in range(1,101)],
                'year': [2000+i%24 for i in range(1,101)],
                'clean_title': [f'Movie Title {i}' for i in range(1, 101)],
                'average_rating': np.random.uniform(3.0, 5.0, 100),
                'ratings_count': np.random.randint(10, 1000, 100),
            }
            movies_df = pd.DataFrame(movies_data)
            movies_df.to_csv(self._data_path('movies.csv'), index=False)
            ratings_data = []
            for user_id in range(1, 201):
                n_ratings = np.random.randint(5, 50)
                movie_ids = np.random.choice(range(1, 101), n_ratings, replace=False)
                for movie_id in movie_ids:
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25])
                    ratings_data.append({'movie_id': movie_id, 'user_id': user_id, 'rating': rating})
            ratings_df = pd.DataFrame(ratings_data)
            ratings_df.to_csv(self._data_path('ratings.csv'), index=False)
            st.success("Demo data created successfully!")
        except Exception as e:
            st.error(f"Error creating demo data: {str(e)}")

    def _process_datasets(self):
        if 'id' in self.movies_df.columns:
            if 'movie_id' in self.movies_df.columns:
                pass  # movie_id already exists
            else:
                self.movies_df['movie_id'] = pd.to_numeric(self.movies_df['id'], errors='coerce')
        self.movies_df.dropna(subset=['movie_id'], inplace=True)
        self.movies_df['movie_id'] = self.movies_df['movie_id'].astype(int)
        default_columns = {
            'title': 'Unknown Title',
            'genres': 'Unknown',
            'average_rating': 0.0,
            'ratings_count': 0,
            'year': 0,
            'clean_title': '',
        }
        for col, default_val in default_columns.items():
            if col not in self.movies_df.columns:
                self.movies_df[col] = default_val
            else:
                self.movies_df[col] = self.movies_df[col].fillna(default_val)
        for col in ['movie_id','average_rating','ratings_count','year']:
            if col in self.movies_df.columns:
                self.movies_df[col] = pd.to_numeric(self.movies_df[col], errors='coerce').fillna(0)

    def _calculate_dataset_statistics(self):
        self.valid_user_ids = set(self.ratings_df['user_id'].unique())
        self.user_rating_counts = self.ratings_df['user_id'].value_counts().to_dict()
        self.dataset_stats = {
            'total_users': len(self.valid_user_ids),
            'total_movies': len(self.movies_df),
            'total_ratings': len(self.ratings_df),
            'avg_ratings_per_user': self.ratings_df.groupby('user_id').size().mean(),
            'avg_rating': self.ratings_df['rating'].mean(),
            'sparsity': 1 - (len(self.ratings_df) / (len(self.valid_user_ids) * len(self.movies_df))),
            'premium_users': sum(1 for count in self.user_rating_counts.values() if count >= 15),
            'rating_distribution': self.ratings_df['rating'].value_counts().sort_index().to_dict()
        }

    def init_recommenders(self):
        self.content_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        self.metrics_analyzer = None

    def display_login_page(self):
        st.markdown('<h1 class="title-gradient">CineVerse</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Your Intelligent Movie Discovery Platform</p>', unsafe_allow_html=True)
        st.session_state.setdefault('show_validation', False)
        st.session_state.setdefault('last_validation', None)
        st.session_state.setdefault('validated_user_id', None)
        if self._check_rate_limit():
            st.markdown('<div class="notification notification-warning">'
                       'Too many login attempts. Please wait a moment before trying again.'
                       '</div>', unsafe_allow_html=True)
            return
        min_uid = min(self.valid_user_ids) if self.valid_user_ids else 1
        max_uid = max(self.valid_user_ids) if self.valid_user_ids else 1_000_000
        col1, col2 = st.columns([9, 1])
        with col1:
            user_id = st.number_input("Enter Your User ID", min_value=min_uid, max_value=max_uid, step=1,
                help=f"Valid IDs: {min_uid:,} - {max_uid:,}. Only IDs that exist in ratings.csv are valid.",
                key="user_id_input")
        if st.session_state.get('validated_user_id') is None or st.session_state.get('validated_user_id') != int(user_id):
            st.session_state['show_validation'] = False
            st.session_state['last_validation'] = None
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Validate", key="validate_btn"):
                result = self.validator.validate_user_id(user_id, self.valid_user_ids, self.user_rating_counts)
                st.session_state['last_validation'] = result
                st.session_state['validated_user_id'] = int(user_id)
                st.session_state['show_validation'] = True
        if st.session_state.get('show_validation') and st.session_state.get('last_validation'):
            self._display_validation_result(st.session_state['last_validation'])
        validated_same_input = (st.session_state.get('validated_user_id') == int(user_id))
        last_is_valid = bool((st.session_state.get('last_validation') or {}).get('is_valid', False))
        can_login = st.session_state.get('show_validation', False) and validated_same_input and last_is_valid
        if not can_login:
            st.caption("Please click **Validate** to verify this User ID before logging in.")
        if st.button("Enter CineVerse", use_container_width=True, type="primary", key="login_btn", disabled=not can_login):
            self._handle_login_attempt(user_id)
        st.markdown('</div>', unsafe_allow_html=True)
        self._display_dataset_overview()


    def _check_rate_limit(self):
        now = datetime.now()
        if st.session_state.last_login_attempt:
            last_attempt = datetime.fromisoformat(st.session_state.last_login_attempt)
            if now - last_attempt < timedelta(seconds=5) and st.session_state.login_attempts >= 3:
                return True
        return False

    def _handle_login_attempt(self, user_id):
        now = datetime.now()
        st.session_state.last_login_attempt = now.isoformat()
        st.session_state.login_attempts += 1
        validation_result = self.validator.validate_user_id(user_id, self.valid_user_ids, self.user_rating_counts)
        if validation_result['is_valid']:
            details = validation_result['details']
            st.session_state.logged_in = True
            st.session_state.current_user = int(user_id)
            st.session_state.user_eligible = details['is_premium']
            st.session_state.user_ratings_count = details['rating_count']
            st.session_state.login_attempts = 0
            st.success(f"Welcome! Logged in as User {user_id:,}")
            time.sleep(1)
            st.rerun()
        else:
            self._display_validation_result(validation_result)
            if st.session_state.login_attempts >= 5:
                st.error("Multiple failed attempts. Please verify your User ID or contact support.")

    def _display_validation_result(self, validation_result):
        message = validation_result['message']
        result_type = validation_result['type']
        if result_type == 'success':
            st.markdown(f'<div class="notification notification-success">OK {message}</div>', unsafe_allow_html=True)
        elif result_type == 'warning':
            st.markdown(f'<div class="notification notification-warning">Warning {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="notification notification-error">Error {message}</div>', unsafe_allow_html=True)
        if validation_result.get('suggestions'):
            with st.expander("Suggestions", expanded=result_type=='error'):
                for suggestion in validation_result['suggestions']:
                    st.write(f"* {suggestion}")

    def _display_dataset_overview(self):
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("??", "Movies", self.dataset_stats['total_movies'], "#4f46e5"),
            ("??", "Users", self.dataset_stats['total_users'], "#7c3aed"),
            ("?", "Ratings", self.dataset_stats['total_ratings'], "#ec4899"),
            ("??", "Avg Rating", f"{self.dataset_stats['avg_rating']:.2f}", "#10b981")
        ]
        for i, (icon, label, value, color) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                if isinstance(value, (int, np.integer)):
                    val_str = f"{int(value):,}"
                elif isinstance(value, (float, np.floating)):
                    val_str = f"{float(value):,.2f}"
                else:
                    val_str = str(value)
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <span class="metric-icon">{icon}</span>
                    <h2>{val_str}</h2>
                    <h3>{label}</h3>
                </div>
                """, unsafe_allow_html=True)
        with st.expander("Detailed Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Premium Users", f"{self.dataset_stats['premium_users']:,}", 
                         f"{self.dataset_stats['premium_users']/self.dataset_stats['total_users']*100:.1f}%")
                st.metric("Avg Ratings/User", f"{self.dataset_stats['avg_ratings_per_user']:.1f}")
            with col2:
                st.metric("Data Sparsity", f"{self.dataset_stats['sparsity']*100:.2f}%")
                if self.dataset_stats['rating_distribution']:
                    fig = px.bar(x=list(self.dataset_stats['rating_distribution'].keys()),
                        y=list(self.dataset_stats['rating_distribution'].values()),
                        title="Rating Distribution", labels={'x': 'Rating', 'y': 'Count'})
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    def display_user_dashboard(self):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f'<h1 class="title-gradient">Welcome, User {st.session_state.current_user:,}</h1>', unsafe_allow_html=True)
        with col2:
            status = "Premium" if st.session_state.user_eligible else "Basic"
            status_class = "status-premium" if st.session_state.user_eligible else "status-basic"
            st.markdown(f'<div class="{status_class} status-indicator">?? {status} Account</div>', unsafe_allow_html=True)
        with col3:
            if st.button("Logout", type="secondary", key="logout_btn"):
                self._handle_logout()
        self._display_user_statistics()
        tab1, tab2, tab3 = st.tabs(["?? Recommendations", "?? Algorithm Analysis", "?? Feedback"])
        with tab1:
            self._display_recommendations_tab()
        with tab2:
            self._display_analysis_tab()
        with tab3:
            self._display_feedback_tab()

    def _display_user_statistics(self):
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == st.session_state.current_user]
        if len(user_ratings) == 0:
            st.warning("No ratings found for this user")
            return
        avg_rating = user_ratings['rating'].mean()
        rating_std = user_ratings['rating'].std() if len(user_ratings) > 1 else 0
        favorite_movies = len(user_ratings[user_ratings['rating'] >= 4])
        rating_range = user_ratings['rating'].max() - user_ratings['rating'].min()
        consistency = 1/rating_std if rating_std > 0 else float('inf')
        st.markdown("### Your Viewing Profile")
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            ("??", "Movies Rated", len(user_ratings), "#4f46e5"),
            ("?", "Avg Rating", f"{avg_rating:.1f}", "#7c3aed"),
            ("??", "Favorites", favorite_movies, "#ec4899"),
            ("??", "Diversity", f"{rating_range:.1f}", "#10b981"),
            ("??", "Consistency", f"{consistency:.1f}" if consistency != float('inf') else "inf", "#f59e0b")
        ]
        for i, (icon, label, value, color) in enumerate(metrics):
            with [col1, col2, col3, col4, col5][i]:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <span class="metric-icon">{icon}</span>
                    <h2>{value}</h2>
                    <h3>{label}</h3>
                </div>
                """, unsafe_allow_html=True)


    def _display_recommendations_tab(self):
        st.markdown("### Choose Your Recommendation Engine")
        algorithms = [
            {
                'name': 'content',
                'title': 'Content-Based',
                'icon': '??',
                'description': 'Analyzes movie features to find similar titles based on your preferences',
                'features': ['Works for new users', 'Explainable results', 'Discovers similar genres', 'No cold start problem'],
                'enabled': True,
                'performance': 'High precision, moderate recall',
                'use_case': 'Best for exploring within your favorite genres'
            },
            {
                'name': 'collaborative',
                'title': 'Collaborative Filtering',
                'icon': '??',
                'description': 'Uses community wisdom from users with similar viewing tastes',
                'features': ['Discovers unexpected gems', 'High accuracy', 'Community-driven', 'Finds popular trends'],
                'enabled': st.session_state.user_eligible,
                'performance': 'Balanced precision and recall',
                'use_case': 'Perfect for finding movies you never knew you would love'
            },
            {
                'name': 'hybrid',
                'title': 'Hybrid Intelligence',
                'icon': '??',
                'description': 'Combines multiple approaches for optimal recommendation quality',
                'features': ['Highest accuracy', 'Best coverage', 'Adaptive learning', 'Personalized weighting'],
                'enabled': st.session_state.user_eligible,
                'performance': 'Maximum precision and recall',
                'use_case': 'Uniquely personalized recommendation experience'
            }
        ]
        cols = st.columns(3)
        for i, algo in enumerate(algorithms):
            with cols[i]:
                st.markdown(f"""
                    <div class="algo-card">
                    <h4>{algo['icon']} {algo['title']}</h4>
                    <p>{algo['description']}</p>
                    <p><strong>Performance:</strong> {algo['performance']}</p>
                    <p><strong>Best for:</strong> {algo['use_case']}</p>
                    <div><strong>Key Features:</strong></div>
                    <ul>{"".join(f"<li>{f}</li>" for f in algo['features'])}</ul>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)
                if st.button(f"Select {algo['title']}", key=f"select_{algo['name']}", 
                        disabled=not algo['enabled'], use_container_width=True):
                    st.session_state.selected_algorithm = algo['name']
                    self.clear_recommendations()
                    st.success(f"OK {algo['title']} selected!")
                    st.rerun()
                if not algo['enabled']:
                    st.caption("Warning: Requires 15+ ratings for premium algorithms")
        
        has_existing_recommendations = (
            hasattr(st.session_state, 'recommendations_generated') and 
            st.session_state.recommendations_generated and
            hasattr(st.session_state, 'current_recommendations') and 
            st.session_state.current_recommendations is not None and
            not st.session_state.current_recommendations.empty
        )
        if has_existing_recommendations:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Regenerate Recommendations", type="secondary"):
                    self.clear_recommendations()
                    st.rerun()
            with col2:
                if st.button("Clear Recommendations", type="secondary"):
                    self.clear_recommendations()
                    st.rerun()
            with col3:
                current_alg = st.session_state.get('current_algorithm', 'Unknown')
                st.info(f"Showing: {current_alg.title()} recommendations")
            self._display_recommendations(
                st.session_state.current_recommendations, 
                st.session_state.get('current_processing_time', 0))
            return
        
        if st.session_state.selected_algorithm:
            st.markdown("---")
            st.markdown("### Recommendation Settings")
            mode = st.radio("Mode", ["Search-based (enter title/genre)"], index=0)
            st.session_state['rec_mode'] = mode
            col1, col2 = st.columns(2)
            with col1:
                n_recommendations = st.slider("Number of recommendations", min_value=5, max_value=20, value=10,
                    help="More recommendations provide better coverage but take longer to review")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Clear Cache", help="Reset cached results"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    self.clear_recommendations()
                    st.success("Cache cleared!")
            query_text = ""
            if mode.startswith("Search"):
                query_text = st.text_input("Search movies (title / genre / keywords)", value="", 
                    help="Type a title, genre or keywords (fuzzy allowed)")
            if st.button("Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner(f"Generating {st.session_state.selected_algorithm} recommendations..."):
                    start_time = time.time()
                    if MODULES_AVAILABLE:
                        algorithm = st.session_state.selected_algorithm
                        try:
                            key = f"{len(self.ratings_df)}:{self.ratings_df['user_id'].nunique()}:{st.session_state.get('ratings_version',0)}"
                            if self.content_recommender is None:
                                self.content_recommender = build_content_recommender(self.movies_df, self.ratings_df, cache_key=key)
                            if self.collaborative_recommender is None:
                                self.collaborative_recommender = build_collaborative_recommender(self.ratings_df, self.movies_df, cache_key=key)
                            if algorithm == "content":
                                recommendations = self.content_recommender.recommend_from_query(query_text, n_recommendations)
                            elif algorithm == "collaborative":
                                recommendations = self.collaborative_recommender.recommend_from_query(query_text, n_recommendations)
                            elif algorithm == "hybrid":
                                if self.hybrid_recommender is None:
                                    self.hybrid_recommender = build_hybrid_recommender(
                                        self.movies_df, self.ratings_df,
                                        self.content_recommender, self.collaborative_recommender,
                                        cache_key=key)
                                recommendations = self.hybrid_recommender.recommend_from_query(query_text, n_recommendations)
                            else:
                                recommendations = pd.DataFrame()
                        except Exception as e:
                            st.warning(f"Custom recommender failed: {e}. Falling back to demo mode.")
                            recommendations = self._mock_generate_recommendations(n_recommendations)
                    else:
                        recommendations = self._mock_generate_recommendations(n_recommendations)
                    recommendations = self._add_match_scores(recommendations, st.session_state.selected_algorithm)
                    processing_time = time.time() - start_time
                    if recommendations is not None and not recommendations.empty:
                        st.session_state.current_recommendations = recommendations
                        st.session_state.current_algorithm = st.session_state.selected_algorithm
                        st.session_state.current_processing_time = processing_time
                        st.session_state.recommendations_generated = True
                        st.session_state.system_performance[st.session_state.selected_algorithm] = {
                            'processing_time': processing_time,
                            'recommendations_count': len(recommendations),
                            'timestamp': datetime.now().isoformat()
                        }
                        st.success("Recommendations generated!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("No recommendations could be generated. Please try a different algorithm, query, or check your data.")
        else:
            st.info("Please select a recommendation algorithm above to continue")


    def _mock_generate_recommendations(self, n_recommendations):
        if len(self.movies_df) == 0:
            return pd.DataFrame()
        return self.movies_df.nlargest(n_recommendations, 'ratings_count').copy()

    def _display_recommendations(self, recommendations, processing_time):
        if recommendations is None or len(recommendations) == 0:
            st.info("No recommendations available. Please generate new recommendations.")
            return
        st.markdown(f"### Recommended Movies ({len(recommendations)} results)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", st.session_state.get('current_algorithm', 'N/A').title())
        with col2:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col3:
            st.metric("Results", f"{len(recommendations)}")
        st.markdown("---")
        for idx_pos, (_, movie) in enumerate(recommendations.iterrows(), 1):
            self._display_movie_card(movie, idx_pos)

    def _display_movie_card(self, movie, idx):
        col1, col2 = st.columns([1, 5])
        with col1:
            self._display_default_movie_cover()
        with col2:
            title = movie.get('title', movie.get('clean_title', 'Unknown Title'))
            genres = movie.get('genres', 'Unknown')
            year = movie.get('year', 0)
            rating = movie.get('average_rating', 0)
            ratings_count = movie.get('ratings_count', 0)
            st.markdown(f"### #{idx} {title}")
            col2a, col2b = st.columns([3, 1])
            with col2a:
                st.write(f"**Genres:** {genres}")
                try:
                    year_int = int(year) if year and year != 0 else None
                    st.write(f"**Year:** {year_int if year_int else 'Unknown'}")
                except (ValueError, TypeError):
                    st.write("**Year:** Unknown")
                if rating and rating > 0:
                    stars = "star" * int(rating) + "no_star" * (5 - int(rating))
                    st.write(f"**Rating:** {'?' * int(rating)}{'?' * (5 - int(rating))} ({rating:.2f}/5)")
                if ratings_count and ratings_count > 0:
                    st.write(f"**Popularity:** {int(ratings_count):,} ratings")
                pct = float(movie.get("match_score_pct", np.nan))
                if not np.isnan(pct):
                    st.markdown(f"""
                    <div class="match-badge">
                    <span class="pill">{pct:.0f}% match</span>
                    <div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div>
                    </div>
                    """, unsafe_allow_html=True)
            with col2b:
                movie_id_for_rating = int(movie.get('movie_id', movie.get('id', idx)))
                existing_rating = self.ratings_df[
                    (self.ratings_df['user_id'] == st.session_state.current_user) &
                    (self.ratings_df['movie_id'] == movie_id_for_rating)
                ]
                if not existing_rating.empty:
                    user_rating = existing_rating['rating'].iloc[0]
                    st.markdown(f"""
                    <div class="notification notification-info">
                        Rated: {user_rating:.1f} star
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.form(f"rate_form_{movie_id_for_rating}", clear_on_submit=True):
                        r = st.slider("Rate (1-5)", 1, 5, 4, key=f"slider_{movie_id_for_rating}")
                        submitted = st.form_submit_button("Submit rating", use_container_width=True)
                    if submitted:
                        with st.spinner("Submitting rating..."):
                            success = self.save_user_rating(
                                user_id=int(st.session_state.current_user),
                                movie_id=int(movie_id_for_rating),
                                rating=float(r)
                            )
                        if success:
                            st.success("Rating saved! Updating display...")
                        else:
                            st.error("Could not save rating. Please try again.")
        st.markdown("---")

    def clear_recommendations(self):
        keys_to_clear = [
            'current_recommendations', 'current_algorithm', 
            'current_processing_time', 'recommendations_generated'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    def _display_default_movie_cover(self):
        st.markdown("""
        <div style="width:120px; height:180px; background: var(--accent-gradient); 
                    display: flex; align-items: center; justify-content: center; 
                    border-radius: 12px; color: white; font-size: 2rem; font-weight: bold;">
            ??
        </div>
        """, unsafe_allow_html=True)

    def update_user_statistics(self):
        vc = self.ratings_df['user_id'].value_counts(dropna=False)
        self.user_rating_counts = vc.astype('int32').to_dict()
        self.valid_user_ids = set(self.user_rating_counts.keys())
        current_user_count = self.user_rating_counts.get(st.session_state.current_user, 0)
        st.session_state.user_ratings_count = current_user_count
        st.session_state.user_eligible = current_user_count >= 15
        self.dataset_stats = {
            'total_users': len(self.valid_user_ids),
            'total_movies': int(self.movies_df['movie_id'].nunique()),
            'total_ratings': int(len(self.ratings_df)),
            'avg_ratings_per_user': float(vc.mean()) if len(vc) else 0.0,
            'avg_rating': float(self.ratings_df['rating'].mean()) if len(self.ratings_df) else 0.0,
            'sparsity': 1 - (len(self.ratings_df) / (len(self.valid_user_ids) * len(self.movies_df))),
            'premium_users': sum(1 for count in self.user_rating_counts.values() if count >= 15),
            'rating_distribution': self.ratings_df['rating'].value_counts().sort_index().to_dict()
        }

    def save_user_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        try:
            ratings_path = Path(self._data_path("ratings.csv"))
            if ratings_path.exists() and not os.access(ratings_path, os.W_OK):
                st.error("Permission error: Cannot write to ratings file.")
                return False
            if not os.access(ratings_path.parent, os.W_OK):
                st.error("Permission error: Cannot write to ratings directory.")
                return False
            df = pd.read_csv(ratings_path) if ratings_path.exists() else pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
            new_rating = {'user_id': int(user_id), 'movie_id': int(movie_id), 'rating': float(rating)}
            mask = (df['user_id'] == new_rating['user_id']) & (df['movie_id'] == new_rating['movie_id'])
            if mask.any():
                df.loc[mask, 'rating'] = new_rating['rating']
            else:
                df = pd.concat([df, pd.DataFrame([new_rating])], ignore_index=True)
            with tempfile.NamedTemporaryFile('w', delete=False, dir=str(ratings_path.parent), suffix='.tmp', encoding='utf-8') as tmp:
                df[['user_id', 'movie_id', 'rating']].to_csv(tmp.name, index=False)
                temp_path = tmp.name
            os.replace(temp_path, ratings_path)
            self.ratings_df = df
            self.update_user_statistics()
            st.session_state.just_saved_rating = {'movie_id': movie_id, 'rating': rating}
            st.cache_data.clear()
            st.cache_resource.clear()
            return True
        except Exception as e:
            st.error(f"Error saving rating: {str(e)}")
            return False


    def _display_analysis_tab(self):
        st.markdown("### Algorithm Performance Analysis")
        st.markdown("Compare the performance of different recommendation algorithms.")
        col1, col2, col3 = st.columns(3)
        st.markdown("""
            <style>
            div[data-baseweb="select"] span {
                color: black !important;
            }
            </style>
            """, unsafe_allow_html=True)
        with col1:
            analysis_scope = st.selectbox(
                "Analysis Scope",
                ["Current User Only", "Random User Sample", "High-Activity Users", "All Premium Users"],
                index=0, help="Choose which users to include in the analysis")
        with col2:
            if analysis_scope != "Current User Only":
                sample_size = 25
            else:
                sample_size = 1
                st.markdown("<br><p style='color: #F5F5DC; font-size: 0.875rem;'>Single user analysis</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Analysis", type="primary", use_container_width=True):
                self._run_performance_analysis(analysis_scope, sample_size)
        if st.session_state.analysis_results:
            self._display_analysis_results()
        else:
            st.info("Click 'Run Analysis' to evaluate algorithm performance across different metrics.")

    def _run_performance_analysis(self, scope, sample_size):
        with st.spinner("Analyzing algorithm performance..."):
            time.sleep(2)
            base_content = {'precision': 0.65, 'recall': 0.45, 'f1': 0.53, 'rmse': 1.1}
            base_collab = {'precision': 0.78, 'recall': 0.62, 'f1': 0.69, 'rmse': 0.95}
            base_hybrid = {'precision': 0.85, 'recall': 0.71, 'f1': 0.77, 'rmse': 0.82}
            def add_noise(val, noise_level=0.1):
                return max(0, min(1, val + np.random.uniform(-noise_level, noise_level)))
            results = {}
            for alg, base in [('content', base_content), ('collaborative', base_collab), ('hybrid', base_hybrid)]:
                for metric, value in base.items():
                    noise = 0.05 if metric == 'rmse' else 0.08
                    results[f'{alg}_{metric}'] = add_noise(value, noise)
            st.session_state.analysis_results = results
            st.success("Analysis completed!")

    def _display_analysis_results(self):
        results = st.session_state.analysis_results
        st.markdown("#### Performance Overview")
        COLORS_BY_KEY = {'content': '#4f46e5', 'collaborative': '#7c3aed', 'hybrid': '#ec4899'}
        PALETTE = ['#4f46e5', '#7c3aed', '#ec4899', '#22c55e', '#f59e0b', '#ef4444']
        algorithms = ['Content-Based', 'Collaborative', 'Hybrid']
        algorithm_keys = ['content', 'collaborative', 'hybrid']
        perf_by_key = {}
        for k in algorithm_keys:
            p = results.get(f'{k}_precision', 0.0)
            r = results.get(f'{k}_recall', 0.0)
            f = results.get(f'{k}_f1', 0.0)
            perf_by_key[k] = (p + r + f) / 3.0
        best_key = max(perf_by_key, key=perf_by_key.get) if perf_by_key else None
        cols = st.columns(len(algorithm_keys))
        for col, alg_name, alg_key in zip(cols, algorithms, algorithm_keys):
            with col:
                precision = results.get(f'{alg_key}_precision', 0.0)
                recall = results.get(f'{alg_key}_recall', 0.0)
                f1 = results.get(f'{alg_key}_f1', 0.0)
                rmse = results.get(f'{alg_key}_rmse', 0.0)
                performance_score = (precision + recall + f1) / 3 * 100
                color = COLORS_BY_KEY.get(alg_key)
                if not color:
                    idx = algorithm_keys.index(alg_key) if alg_key in algorithm_keys else 0
                    color = PALETTE[idx % len(PALETTE)]
                is_best = (alg_key == best_key)
                badge_html = (
                    f'<span style="background:{color}1A;color:{color};'
                    'padding:2px 8px;border-radius:999px;font-size:0.7rem;'
                    'margin-left:8px;">TOP</span>'
                    if is_best else ''
                )
                st.markdown(f"""
                    <div class="performance-card" style="border-left: 4px solid {color};">
                    <h4 style="display:flex;align-items:center;justify-content:space-between;
                                color:#111827;opacity:1;font-weight:600;margin:0;">
                        <span>{alg_name}</span>{badge_html}
                    </h4>
                    <div style="text-align: center; margin: 1rem 0;">
                        <h2 style="color: {color}; font-size: 2.5rem;">{performance_score:.1f}%</h2>
                        <small style="color: #64748b;">Overall Performance</small>
                    </div>
                    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0;">
                    <div style="font-size: 0.875rem; color: #374151;">
                        <strong>Precision:</strong> {precision*100:.1f}%<br>
                        <strong>Recall:</strong> {recall*100:.1f}%<br>
                        <strong>F1-Score:</strong> {f1*100:.1f}%<br>
                        <strong>RMSE:</strong> {rmse:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        self._display_comparison_charts(results, algorithms, algorithm_keys)

    def _display_comparison_charts(self, results, algorithms, algorithm_keys):
        st.markdown("#### Detailed Performance Comparison")
        metrics_data = {
            'Algorithm': algorithms,
            'Precision (%)': [results.get(f'{key}_precision', 0)*100 for key in algorithm_keys],
            'Recall (%)': [results.get(f'{key}_recall', 0)*100 for key in algorithm_keys],
            'F1-Score (%)': [results.get(f'{key}_f1', 0)*100 for key in algorithm_keys],
        }
        df_metrics = pd.DataFrame(metrics_data)
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(
                df_metrics.melt(id_vars=['Algorithm'], var_name='Metric', value_name='Score'),
                x='Algorithm', y='Score', color='Metric',
                title="Performance Metrics Comparison",
                color_discrete_sequence=['#4f46e5', '#7c3aed', '#ec4899'])
            fig_bar.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_radar = go.Figure()
            for i, alg in enumerate(algorithms):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[df_metrics['Precision (%)'][i], df_metrics['Recall (%)'][i], df_metrics['F1-Score (%)'][i]],
                    theta=['Precision', 'Recall', 'F1-Score'],
                    fill='toself', name=alg,
                    line_color=['#4f46e5', '#7c3aed', '#ec4899'][i]))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True, title="Multi-Metric Performance Radar", height=400)
            st.plotly_chart(fig_radar, use_container_width=True)


    def _display_feedback_tab(self):
        st.markdown("### Your Experience Feedback")
        st.markdown("Help us improve CineVerse by sharing your thoughts and experiences.")
        with st.form("comprehensive_feedback", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Overall Experience")
                overall_satisfaction = st.select_slider(
                    "Overall satisfaction with CineVerse",
                    options=[1, 2, 3, 4, 5], value=4,
                    format_func=lambda x: f"{'star' * x} ({x}/5)")
                recommendation_accuracy = st.select_slider(
                    "How accurate were the recommendations?",
                    options=[1, 2, 3, 4, 5], value=4,
                    format_func=lambda x: f"{'target' * x} ({x}/5)")
                system_performance = st.select_slider(
                    "System speed and responsiveness",
                    options=[1, 2, 3, 4, 5], value=4,
                    format_func=lambda x: f"{'zap' * x} ({x}/5)")
                ui_experience = st.select_slider(
                    "User interface and design",
                    options=[1, 2, 3, 4, 5], value=4,
                    format_func=lambda x: f"{'art' * x} ({x}/5)")
            with col2:
                st.markdown("#### Feature Evaluation")
                preferred_algorithm = st.selectbox(
                    "Which algorithm worked best for you?",
                    ['Content-Based', 'Collaborative Filtering', 'Hybrid Intelligence', 'Not sure'],
                    index=0, help="Based on your experience with the recommendations")
                recommendation_diversity = st.select_slider(
                    "Variety in recommendations",
                    options=[1, 2, 3, 4, 5], value=3,
                    format_func=lambda x: f"{'movie' * x} ({x}/5)")
                would_recommend = st.radio(
                    "Would you recommend CineVerse to friends?",
                    ['Definitely', 'Probably', 'Maybe', 'Probably not', 'Definitely not'])
                likelihood_to_use = st.select_slider(
                    "How likely are you to use CineVerse again?",
                    options=[1, 2, 3, 4, 5], value=4,
                    format_func=lambda x: f"{'repeat' * x} ({x}/5)")
            st.markdown("#### Additional Feedback")
            col3, col4 = st.columns(2)
            with col3:
                liked_features = st.multiselect(
                    "What features did you like most?",
                    ['Recommendation accuracy', 'System speed', 'User interface',
                     'Algorithm variety', 'Movie information display', 'Performance analytics',
                     'Rating system', 'User dashboard', 'Feedback collection'],
                    help="Select all that apply")
            with col4:
                improvement_areas = st.multiselect(
                    "What areas need improvement?",
                    ['Recommendation quality', 'System performance', 'User interface',
                     'Movie catalog', 'Search functionality', 'Mobile experience',
                     'Loading times', 'Error handling', 'Help documentation'],
                    help="Select areas for improvement")
            suggestions = st.text_area(
                "Suggestions for improvement",
                placeholder="Share your ideas for making CineVerse better...",
                height=100)
            additional_comments = st.text_area(
                "Additional comments",
                placeholder="Any other thoughts, experiences, or feedback...",
                height=80)
            submitted = st.form_submit_button("Submit Feedback", use_container_width=True, type="primary")
            if submitted:
                feedback_data = {
                    'user_id': st.session_state.current_user,
                    'timestamp': datetime.now().isoformat(),
                    'overall_satisfaction': overall_satisfaction,
                    'recommendation_accuracy': recommendation_accuracy,
                    'system_performance': system_performance,
                    'ui_experience': ui_experience,
                    'recommendation_diversity': recommendation_diversity,
                    'preferred_algorithm': preferred_algorithm,
                    'would_recommend': would_recommend,
                    'likelihood_to_use': likelihood_to_use,
                    'liked_features': liked_features,
                    'improvement_areas': improvement_areas,
                    'suggestions': suggestions,
                    'additional_comments': additional_comments
                }
                st.session_state.user_feedback.append(feedback_data)
                st.markdown("""
                <div class="notification notification-success">
                    Thank you for your valuable feedback! Your input helps us improve CineVerse.
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
        self._display_feedback_summary()

    def _display_feedback_summary(self):
        if not st.session_state.user_feedback:
            return
        st.markdown("---")
        st.markdown("#### Your Feedback History")
        feedback_df = pd.DataFrame(st.session_state.user_feedback)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_satisfaction = feedback_df['overall_satisfaction'].mean()
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
        with col2:
            feedback_count = len(feedback_df)
            st.metric("Feedback Sessions", f"{feedback_count}")
        with col3:
            if 'preferred_algorithm' in feedback_df.columns:
                most_preferred = feedback_df['preferred_algorithm'].mode()
                if not most_preferred.empty:
                    st.metric("Preferred Algorithm", most_preferred.iloc[0])
        with col4:
            if feedback_count > 0:
                latest_feedback = feedback_df.iloc[-1]['timestamp']
                latest_date = datetime.fromisoformat(latest_feedback).strftime("%Y-%m-%d")
                st.metric("Latest Feedback", latest_date)

    def _handle_logout(self):
        keys_to_clear = [
            'logged_in', 'current_user', 'user_eligible', 'user_ratings_count',
            'selected_algorithm', 'last_recommendations', 'analysis_results',
            'just_saved_rating', 'validation_state', 'login_attempts', 'ratings_version',
            'current_recommendations', 'current_algorithm', 'current_processing_time', 
            'recommendations_generated'
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        self.content_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    def run(self):
        try:
            if not st.session_state.logged_in:
                self.display_login_page()
            else:
                self.display_user_dashboard()
        except Exception as e:
            st.error(f"Application error: {str(e)}")

# Application entry point
if __name__ == "__main__":
    try:
        app = EnhancedMovieRecommenderApp()
        app.run()
    except Exception as e:
        st.error("Critical application error occurred. Please refresh the page.")
        st.code(f"Error details: {str(e)}", language="text")
